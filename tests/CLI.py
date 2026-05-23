"""
CLI.py — command-line test interface for the harbour gate system.

Modes
-----
  [1] Full automated entry   — vision → audio → name verify (same flow as GUI)
  [2] Vision only            — run VisionPipeline, print plate + conf, show image
  [3] Audio only             — skip vision, go straight to voice plate request
  [0] Exit

Bugs fixed vs the original
---------------------------
  - read_plate() → run_automated_entry()  (was calling a non-existent method)
  - os.path.join(self.images[choice]) with one arg is a no-op; fixed to join
    images_path + filename correctly
  - create_mock_db() called without db_path arg; now uses config value
  - No result display beyond a raw dict print; replaced with structured output
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, Dict, Any

from scripts.helpers import Orchestrator, read_config, create_mock_db


# ── ANSI colour helpers (no extra dependency) ─────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap `text` in an ANSI colour code if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def yellow(t): return _c(t, "93")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


# ── Result printer ────────────────────────────────────────────────────────────

def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print the dict returned by run_automated_entry."""
    status = result.get("status", "unknown")
    print()
    if status == "success":
        db = result["db_entry"]
        print(green("  ✅  ACCESS GRANTED"))
        print(f"       Driver : {db['driver_name']}")
        print(f"       Plate  : {db['plate']}")
        print(f"       Cargo  : {db['cargo']}")
        print(f"       Dock   : {db['dock']}")
        print(f"       Window : {db['arrival_window']}")
    elif status == "alert_worker":
        print(red("  🚨  ALERT — UNREGISTERED VEHICLE"))
        print(f"       Plate  : {result.get('plate', 'UNKNOWN')}")
        print(red("       Manual intervention required."))
    elif status == "name_mismatch":
        print(red("  🚨  ALERT — NAME MISMATCH"))
        print(f"       Plate  : {result.get('plate', 'UNKNOWN')}")
        print(red("       Driver name does not match records."))
    else:
        print(yellow(f"  ⚠   Unknown status: {status!r}"))
    print()


def print_vision_result(vision_output: Optional[Dict[str, Any]]) -> None:
    """Pretty-print just the vision pipeline output."""
    print()
    if vision_output is None:
        print(yellow("  ⚠   No plate detected by vision model."))
    else:
        conf = vision_output["conf"]
        plate = vision_output["plate"]
        colour = green if conf >= 0.5 else yellow
        print(colour(f"  Plate : {plate}"))
        print(colour(f"  Conf  : {conf:.2f}"))
        print(dim(f"  BBox  : {vision_output['bbox']}"))
    print()


def show_image(image: np.ndarray, window_title: str = "Annotated", timeout_ms: int = 0) -> None:
    """Display an image with cv2. Press any key or wait `timeout_ms` ms to close.
    timeout_ms=0 → wait indefinitely for a keypress.
    """
    cv2.imshow(window_title, image)
    cv2.waitKey(timeout_ms)
    cv2.destroyAllWindows()


# ── CLI class ─────────────────────────────────────────────────────────────────

class CLI:
    """Interactive command-line interface for testing the harbour gate pipeline."""

    def __init__(self, orchestrator: Orchestrator, images_path: str) -> None:
        self.orchestrator = orchestrator
        self.images_path  = images_path
        self._refresh_image_list()

    def _refresh_image_list(self) -> None:
        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(
            f for f in os.listdir(self.images_path)
            if os.path.splitext(f)[1].lower() in supported
        )

    def _choose_image(self) -> Optional[str]:
        """Display a numbered list and return the full path of the chosen image."""
        self._refresh_image_list()
        if not self.images:
            print(yellow("  No images found in: " + self.images_path))
            return None

        print()
        print(bold("  Available images:"))
        for idx, name in enumerate(self.images, start=1):
            print(f"    {dim(str(idx) + '.')} {name}")
        print()

        try:
            choice = int(input("  Select image number: ").strip()) - 1
        except (ValueError, EOFError):
            print(red("  Invalid input."))
            return None

        if 0 <= choice < len(self.images):
            return os.path.join(self.images_path, self.images[choice])
        else:
            print(red("  Number out of range."))
            return None

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path)
        if img is None:
            print(red(f"  Could not read image: {path}"))
        return img

    # ── Modes ─────────────────────────────────────────────────────────────────

    def mode_full_flow(self) -> None:
        """Vision + audio + name verification — identical to the GUI path."""
        path = self._choose_image()
        if not path:
            return

        image = self._load_image(path)
        if image is None:
            return

        print(f"\n  {dim('Running full automated entry flow…')}")
        result, vision_output = self.orchestrator.run_automated_entry(image)

        print_result(result)

        # Show annotated image if vision found anything
        display = vision_output["visual"] if vision_output else image
        show_annotated = input("  Show annotated image? (y/n): ").strip().lower()
        if show_annotated == "y":
            show_image(display, window_title="Gate — Full Flow")

    def mode_vision_only(self) -> None:
        """Run only the vision pipeline, print plate + confidence, optionally show image."""
        path = self._choose_image()
        if not path:
            return

        image = self._load_image(path)
        if image is None:
            return

        print(f"\n  {dim('Running vision pipeline…')}")
        vision_output = self.orchestrator.vision_pipeline.read_plate(image)
        print_vision_result(vision_output)

        if vision_output:
            show_annotated = input("  Show annotated image? (y/n): ").strip().lower()
            if show_annotated == "y":
                show_image(vision_output["visual"], window_title="Gate — Vision Only")

    def mode_audio_only(self) -> None:
        """Skip vision, request the plate verbally, then continue with name verification."""
        print(f"\n  {dim('Starting audio-only plate request…')}")
        audio_result = self.orchestrator.audio_pipeline.request_plate_from_driver(
            vision_output=None
        )
        spoken_plate = audio_result["plate"]
        print(f"\n  Parsed plate : {bold(spoken_plate)}")
        print(f"  Transcription: {dim(audio_result['transcription'])}")

        db_entry = self.orchestrator.lookup_plate(spoken_plate)
        if not db_entry:
            print(red(f"\n  Plate {spoken_plate!r} not found in database."))
            return

        print(green(f"\n  Plate found — {db_entry['driver_name']}, {db_entry['dock']}"))

        # Name verification
        spoken_name = self.orchestrator.audio_pipeline.request_driver_name()
        print(f"\n  Driver said name: {bold(spoken_name)}")
        match = self.orchestrator.audio_pipeline.language_model.verify_name_similarity(
            db_name=db_entry["driver_name"],
            spoken_name=spoken_name,
        )
        if match:
            self.orchestrator.audio_pipeline.give_dock_instructions(db_entry)
            print_result({"status": "success", "db_entry": db_entry})
        else:
            print_result({"status": "name_mismatch", "plate": db_entry["plate"]})

    def mode_batch(self) -> None:
        """Run the full flow on every image in the folder and print a summary table."""
        self._refresh_image_list()
        if not self.images:
            print(yellow("  No images found."))
            return

        print(f"\n  {bold('Batch test')} — {len(self.images)} image(s) in {self.images_path}\n")
        rows = []

        for name in self.images:
            path  = os.path.join(self.images_path, name)
            image = self._load_image(path)
            if image is None:
                rows.append((name, "load error", "-", "-"))
                continue

            result, vision_output = self.orchestrator.run_automated_entry(image)
            status = result.get("status", "?")
            plate  = result.get("plate") or result.get("db_entry", {}).get("plate", "-")
            conf_str = (
                f"{vision_output['conf']:.2f}" if vision_output else "n/a"
            )
            rows.append((name, status, plate, conf_str))

        # Print table
        col_w = [max(len(r[i]) for r in rows + [("File", "Status", "Plate", "Conf")])
                 for i in range(4)]
        header = (
            f"  {'File':<{col_w[0]}}  {'Status':<{col_w[1]}}  "
            f"{'Plate':<{col_w[2]}}  {'Conf':<{col_w[3]}}"
        )
        print(bold(header))
        print("  " + "-" * (sum(col_w) + 8))
        for name, status, plate, conf in rows:
            colour = green if status == "success" else (
                     red if status in ("alert_worker", "name_mismatch") else yellow)
            print(colour(
                f"  {name:<{col_w[0]}}  {status:<{col_w[1]}}  "
                f"{plate:<{col_w[2]}}  {conf:<{col_w[3]}}"
            ))
        print()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print()
        print(bold("  ════════════════════════════════"))
        print(bold("   Harbor Gate  —  CLI Test Tool  "))
        print(bold("  ════════════════════════════════"))

        menu = {
            "1": ("Full automated entry flow", self.mode_full_flow),
            "2": ("Vision only",               self.mode_vision_only),
            "3": ("Audio only",                self.mode_audio_only),
            "4": ("Batch test (all images)",   self.mode_batch),
            "0": ("Exit",                      None),
        }

        while True:
            print()
            for key, (label, _) in menu.items():
                print(f"  [{bold(key)}] {label}")
            print()

            choice = input("  Select mode: ").strip()

            if choice == "0":
                print(dim("  Goodbye.\n"))
                break
            elif choice in menu:
                _, fn = menu[choice]
                try:
                    fn()
                except KeyboardInterrupt:
                    print(yellow("\n  Interrupted."))
            else:
                print(red("  Unknown option."))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = read_config("utils/config.yaml")

    if not os.path.exists(config["database"]["db_path"]):
        create_mock_db(config["database"]["db_path"])   # bug fix: pass db_path arg

    with Orchestrator(config, onnx=True) as orchestrator:
        cli = CLI(orchestrator, config["images"]["path"])
        cli.run()
