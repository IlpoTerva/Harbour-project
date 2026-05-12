import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from typing import Dict, Any, Optional
import logging
import threading
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def log(self, message: str) -> None:
        """Append `message` to the assistant log box (safe to call from any thread)."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert(tk.END, f"> {message}\n\n")
            self.log_box.config(state="disabled")
            self.log_box.see(tk.END)
        self.root.after(0, _write)

    def display_image(self, image: np.ndarray) -> None:
        """Resize and show a BGR numpy image in the image panel."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # keep a reference

    def set_status(self, text: str, colour: str) -> None:
        """Update the status label (safe to call from any thread)."""
        self.root.after(0, lambda: self.result_label.config(text=text, fg=colour))

    def set_button_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.root.after(0, lambda: self.import_btn.config(state=state))

    # ── Main event handlers ───────────────────────────────────────────────────

    def import_image(self) -> None:
        image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not image_path:
            return

        raw_image = cv2.imread(image_path)
        result, vision_output = self.orchestrator.read_plate(raw_image)
        status = result["status"]

        # Always display the annotated image when available, raw image otherwise.
        display_image = vision_output["visual"] if vision_output else raw_image
        self.display_image(display_image)

        if status == "no_detection":
            self.set_status("NO PLATE DETECTED", "red")
            self.log("No licence plate found in the image.")
            self.set_button_enabled(False)
            threading.Thread(
                target=self._audio_fallback_worker,
                args=(vision_output,),
                daemon=True,
            ).start()

        elif status == "recognized":
            db = result["db_entry"]
            plate = result["plate"]
            conf = result["conf"]
            self.set_status(f"ENTRY PERMITTED: {plate} ({conf:.2f})", "green")
            self.log(
                f"Plate {plate} identified (conf {conf:.2f}).\n"
                f"  Cargo: {db['cargo']}\n"
                f"  Dock:  {db['dock']}\n"
                f"  Window:{db['arrival_window']}"
            )

        elif status == "not_in_db":
            plate = result["plate"]
            conf = result["conf"]
            self.set_status(f"PLATE NOT IN DB: {plate} ({conf:.2f})", "orange")
            self.log(f"Plate {plate!r} read with conf {conf:.2f} but is not in the manifest.")

        elif status == "low_confidence":
            plate = result.get("plate", "?")
            conf = result.get("conf", 0)
            self.set_status(f"LOW CONF ({conf:.2f}) — LISTENING…", "orange")
            self.log(f"Read {plate!r} with low confidence ({conf:.2f}). Starting voice fallback…")
            # Disable the import button while the audio flow is running.
            self.set_button_enabled(False)
            threading.Thread(
                target=self._audio_fallback_worker,
                args=(vision_output,),
                daemon=True,
            ).start()

    def _audio_fallback_worker(self, vision_output: Optional[Dict[str, Any]]) -> None:
        """Run the audio fallback in a background thread, then update the GUI."""
        try:
            db_entry, spoken_plate = self.orchestrator.handle_audio_fallback(vision_output)
        except Exception:
            logger.exception("Audio fallback failed.")
            self.root.after(0, self._on_audio_error)
            return
        self.root.after(0, self._on_audio_result, db_entry, spoken_plate)

    def _on_audio_result(
        self, db_entry: Optional[Dict[str, Any]], spoken_plate: str
    ) -> None:
        """Called on the main thread once the audio fallback has finished."""
        self.set_button_enabled(True)
        if db_entry:
            self.set_status(f"AUDIO OK: {spoken_plate}", "green")
            self.log(
                f"Driver said {spoken_plate!r} — found in manifest.\n"
                f"  Cargo: {db_entry['cargo']}\n"
                f"  Dock:  {db_entry['dock']}\n"
                f"  Window:{db_entry['arrival_window']}"
            )
        else:
            self.set_status(f"AUDIO: {spoken_plate} NOT IN DB", "red")
            self.log(f"Driver said {spoken_plate!r} — not found in the manifest. Manual check required.")

    def _on_audio_error(self) -> None:
        self.set_button_enabled(True)
        self.set_status("AUDIO ERROR", "red")
        self.log("Voice fallback failed. Please check microphone and try again.")

    def run(self) -> None:
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = read_config("utils/config.yaml")
    if not os.path.exists(config["database"]["db_path"]):
        create_mock_db()
    
    with Orchestrator(config, onnx=True) as orchestrator:
        gui = GUI(orchestrator)
        gui.run()