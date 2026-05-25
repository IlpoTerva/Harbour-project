"""
Laptop frontend for the Harbour Agent.

The heavy ML work (YOLOv11, Fast-Plate-OCR, Faster-Whisper, Piper TTS,
Llama 3.2 1B, SQLite) runs on the Jetson Orin NX via scripts/api_server.py.
This module handles everything on the laptop side:

  - HarbourClient   — thin HTTP wrapper around every server endpoint
  - RemoteOrchestrator — full gate-flow logic (mirrors scripts/orchestrator.py)
                         using HarbourClient for ML/DB and sounddevice for audio I/O
  - RemoteGUI       — Tkinter interface (mirrors UI/GUI.py; no ML imports)

Run (from the project root on the laptop):
    python UI/remote_frontend.py --host http://<jetson-ip>:8000

Dependencies needed on the laptop (no ML packages required):
    pip install requests sounddevice numpy opencv-python Pillow
"""

import argparse
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from client import HarbourClient
from remote_orchestrator import RemoteOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Tkinter GUI ───────────────────────────────────────────────────────────────

class RemoteGUI:
    """
    Tkinter frontend for the laptop.  Functionally identical to UI/GUI.py but
    carries no imports from scripts/ so it can run without ML packages installed.
    """

    DISPLAY_SIZE = (400, 300)

    def __init__(self, orchestrator: RemoteOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.orchestrator.on_vision_result = self._on_vision_update

        self.root = tk.Tk()
        self.root.title("Harbor Gate: AI Logistics System (Remote)")

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

        self.db_btn = tk.Button(
            left,
            text="VIEW DATABASE",
            command=self.open_db_window,
            bg="#27ae60",
            fg="white",
            font=("Helvetica", 12, "bold"),
            height=2,
            width=25,
        )
        self.db_btn.pack(pady=(0, 10))

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
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert(tk.END, f"> {message}\n\n")
            self.log_box.config(state="disabled")
            self.log_box.see(tk.END)
        self.root.after(0, _write)

    def set_status(self, text: str, colour: str) -> None:
        self.root.after(0, lambda: self.result_label.config(text=text, fg=colour))

    def set_button_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.root.after(0, lambda: self.import_btn.config(state=state))

    def display_image(self, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

    # ── Vision callback ───────────────────────────────────────────────────────

    def _on_vision_update(self, vision_output: Optional[Dict[str, Any]]) -> None:
        """Called from the worker thread as soon as plate detection returns."""
        if vision_output is not None:
            self.root.after(0, self.display_image, vision_output["visual"])

    # ── Event handlers ────────────────────────────────────────────────────────

    def import_image(self) -> None:
        image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not image_path:
            return

        raw_image = cv2.imread(image_path)
        self.display_image(raw_image)
        self.set_button_enabled(False)
        self.set_status("PROCESSING…", "orange")
        self.log("Entry flow started. Running vision and voice verification…")

        threading.Thread(
            target=self._flow_worker,
            args=(raw_image,),
            daemon=True,
        ).start()

    def open_db_window(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Truck Database")
        popup.configure(bg="#2c3e50")
        popup.resizable(True, True)

        tk.Label(
            popup,
            text="Registered Trucks",
            font=("Helvetica", 14, "bold"),
            bg="#2c3e50",
            fg="white",
        ).pack(pady=(12, 4))

        status_var = tk.StringVar(value="Loading…")
        status_label = tk.Label(
            popup,
            textvariable=status_var,
            font=("Helvetica", 10),
            bg="#2c3e50",
            fg="white",
        )
        status_label.pack()

        frame = tk.Frame(popup, bg="#2c3e50")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        columns = ("plate", "driver_name", "cargo", "dock", "arrival_window")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)

        col_labels = {
            "plate":          "Plate",
            "driver_name":    "Driver Name",
            "cargo":          "Cargo",
            "dock":           "Dock",
            "arrival_window": "Arrival Window",
        }
        col_widths = {
            "plate":          90,
            "driver_name":    150,
            "cargo":          120,
            "dock":           70,
            "arrival_window": 110,
        }
        for col in columns:
            tree.heading(col, text=col_labels[col])
            tree.column(col, width=col_widths[col], anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _fetch():
            try:
                rows = self.orchestrator.client.list_all_plates()
                def _populate():
                    for item in tree.get_children():
                        tree.delete(item)
                    for row in rows:
                        tree.insert(
                            "",
                            tk.END,
                            values=(
                                row["plate"],
                                row["driver_name"],
                                row["cargo"],
                                row["dock"],
                                row["arrival_window"],
                            ),
                        )
                    status_var.set(f"{len(rows)} record(s) loaded.")
                self.root.after(0, _populate)
            except Exception as exc:
                self.root.after(0, lambda: status_var.set(f"Error: {exc}"))

        def _refresh():
            status_var.set("Loading…")
            threading.Thread(target=_fetch, daemon=True).start()

        tk.Button(
            popup,
            text="Refresh",
            command=_refresh,
            bg="#27ae60",
            fg="white",
            font=("Helvetica", 11, "bold"),
            width=12,
        ).pack(pady=(4, 12))

        threading.Thread(target=_fetch, daemon=True).start()

    def _flow_worker(self, raw_image: np.ndarray) -> None:
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
        self.set_button_enabled(True)

        display_img = vision_output["visual"] if vision_output else raw_image
        self.display_image(display_img)

        status = result["status"]

        if status == "success":
            db = result["db_entry"]
            self.set_status(f"ENTRY PERMITTED: {db['plate']}", "green")
            self.log(
                f"Verification complete.\n"
                f"   Driver: {db['driver_name']}\n"
                f"   Cargo:  {db['cargo']}\n"
                f"   Dock:   {db['dock']}\n"
                f"   Window: {db['arrival_window']}"
            )
        elif status == "alert_worker":
            plate = result.get("plate", "UNKNOWN")
            self.set_status("ALERT: UNREGISTERED VEHICLE", "red")
            self.log(
                f"Plate '{plate}' is not in the database.\n"
                "   Manual intervention required."
            )
        elif status == "name_mismatch":
            plate = result.get("plate", "UNKNOWN")
            self.set_status("ALERT: NAME MISMATCH", "red")
            self.log(
                f"Driver name verification failed for '{plate}'.\n"
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
    parser = argparse.ArgumentParser(description="Harbour Agent laptop frontend")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="Jetson API server URL, e.g. http://192.168.1.50:8000",
    )
    args = parser.parse_args()

    client = HarbourClient(base_url=args.host)

    logger.info(f"Connecting to server at {args.host} …")
    try:
        health = client.health()
        logger.info(f"Server health: {health}")
    except Exception as exc:
        logger.error(f"Cannot reach server at {args.host}: {exc}")
        raise SystemExit(1)

    orchestrator = RemoteOrchestrator(client=client)
    gui = RemoteGUI(orchestrator=orchestrator)
    gui.run()
