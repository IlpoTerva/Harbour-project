import os
import logging
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from typing import Dict, Any, Optional

from scripts.orchestrator import Orchestrator, read_config, create_mock_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class GUI:
    """Tkinter front-end for the harbour gate system."""

    DISPLAY_SIZE = (400, 300)

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

        self.root = tk.Tk()
        self.root.title("Harbor Gate: AI Logistics System")

        main = tk.Frame(self.root)
        main.pack(padx=20, pady=20)

        # ── Left column: image + status + button ──────────────────────────────
        left = tk.Frame(main)
        left.pack(side=tk.LEFT, padx=10)

        self.image_frame = tk.Frame(
            left,
            width=self.DISPLAY_SIZE[0],
            height=self.DISPLAY_SIZE[1],
            bg="#2c3e50",
        )
        self.image_frame.pack()
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#2c3e50")
        self.image_label.pack(expand=True)

        self.result_label = tk.Label(
            left, text="System Ready", font=("Helvetica", 16, "bold"), pady=10
        )
        self.result_label.pack()

        self.import_btn = tk.Button(
            left,
            text="IMPORT TRUCK IMAGE",
            command=self.import_image,
            bg="#27ae60",
            fg="white",
            font=("Helvetica", 12, "bold"),
            height=2,
            width=25,
        )
        self.import_btn.pack(pady=10)

        # ── Right column: log box ─────────────────────────────────────────────
        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        tk.Label(right, text="Gate Assistant Logs", font=("Helvetica", 12)).pack()

        self.log_box = scrolledtext.ScrolledText(
            right, width=35, height=20, state="disabled", font=("Consolas", 10)
        )
        self.log_box.pack()

    # ── Thread-safe UI helpers ────────────────────────────────────────────────

    def log(self, message: str) -> None:
        """Append a message to the log box. Safe to call from any thread."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert(tk.END, f"> {message}\n\n")
            self.log_box.config(state="disabled")
            self.log_box.see(tk.END)
        self.root.after(0, _write)

    def set_status(self, text: str, colour: str) -> None:
        """Update the status label. Safe to call from any thread."""
        self.root.after(0, lambda: self.result_label.config(text=text, fg=colour))

    def set_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the import button. Safe to call from any thread."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.root.after(0, lambda: self.import_btn.config(state=state))

    def display_image(self, image: np.ndarray) -> None:
        """Resize and show a BGR numpy image in the image panel."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # keep reference to prevent GC

    # ── Main event handlers ───────────────────────────────────────────────────

    def import_image(self) -> None:
        image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not image_path:
            return

        raw_image = cv2.imread(image_path)

        # Show the raw image immediately so the screen isn't blank during processing
        self.display_image(raw_image)
        self.set_button_enabled(False)
        self.set_status("PROCESSING…", "orange")
        self.log("Entry flow started. Running vision and voice verification…")

        threading.Thread(
            target=self._flow_worker,
            args=(raw_image,),
            daemon=True,
        ).start()

    def _flow_worker(self, raw_image: np.ndarray) -> None:
        """Runs the full orchestrator flow in a background thread.

        run_automated_entry always returns (result, vision_output) so the
        unpack here is guaranteed safe.
        """
        try:
            result, vision_output = self.orchestrator.run_automated_entry(raw_image)
        except Exception as e:
            logger.exception("Orchestrator flow raised an exception.")
            self.root.after(0, self._on_error, str(e))
            return
        self.root.after(0, self._on_complete, result, vision_output, raw_image)

    def _on_complete(
        self,
        result: Dict[str, Any],
        vision_output: Optional[Dict[str, Any]],
        raw_image: np.ndarray,
    ) -> None:
        """Called on the main thread once the orchestrator flow finishes."""
        self.set_button_enabled(True)

        # Show annotated image if YOLO found anything, otherwise keep the raw image
        display_img = vision_output["visual"] if vision_output else raw_image
        self.display_image(display_img)

        status = result["status"]

        if status == "success":
            db = result["db_entry"]
            self.set_status(f"ENTRY PERMITTED: {db['plate']}", "green")
            self.log(
                f"✅ Verification complete.\n"
                f"   Driver: {db['driver_name']}\n"
                f"   Cargo:  {db['cargo']}\n"
                f"   Dock:   {db['dock']}\n"
                f"   Window: {db['arrival_window']}"
            )

        elif status == "alert_worker":
            plate = result.get("plate", "UNKNOWN")
            self.set_status("🚨 ALERT: UNREGISTERED VEHICLE", "red")
            self.log(
                f"🚨 Plate '{plate}' is not in the database.\n"
                "   Manual intervention required."
            )

        elif status == "name_mismatch":
            plate = result.get("plate", "UNKNOWN")
            self.set_status("🚨 ALERT: NAME MISMATCH", "red")
            self.log(
                f"🚨 Driver name verification failed for '{plate}'.\n"
                "   Manual intervention required."
            )

        else:
            self.set_status("UNKNOWN STATUS", "orange")
            self.log(f"Unexpected status: {status!r}")

    def _on_error(self, err_msg: str) -> None:
        self.set_button_enabled(True)
        self.set_status("SYSTEM ERROR", "red")
        self.log(f"Fatal error: {err_msg}")

    def run(self) -> None:
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = read_config("utils/config.yaml")
    if not os.path.exists(config["database"]["db_path"]):
        create_mock_db(config["database"]["db_path"])
    with Orchestrator(config, onnx=True) as orchestrator:
        gui = GUI(orchestrator)
        gui.run()
