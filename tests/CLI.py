"""
CLI.py — command-line test interface for the harbour gate system.

Connects to the running Jetson API server via HarbourClient and exercises
the full gate pipeline through RemoteOrchestrator.

Modes
-----
  [1] Full automated entry   — vision → audio → name verify (same flow as GUI)
  [2] Vision only            — call /vision/detect, print plate + conf, show image
  [3] Audio only             — skip vision, go straight to voice plate request
  [4] Batch test             — run full flow on every image in the folder
  [0] Exit

Run from the project root:
    python tests/CLI.py [--host http://<jetson-ip>:8000]
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, Any

# Allow bare imports used inside remote_orchestrator.py and client.py
sys.path.insert(0, str(Path(__file__).parent.parent / "UI"))

from client import HarbourClient
from remote_orchestrator import RemoteOrchestrator
from scripts.helpers import read_config


# ── ANSI colour helpers ───────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def yellow(t): return _c(t, "93")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


# ── Result printers ───────────────────────────────────────────────────────────

def print_result(result: Dict[str, Any]) -> None:
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
    cv2.imshow(window_title, image)
    cv2.waitKey(timeout_ms)
    cv2.destroyAllWindows()


# ── CLI class ─────────────────────────────────────────────────────────────────

class CLI:
    """Interactive command-line interface for testing the harbour gate pipeline."""

    def __init__(self, orchestrator: RemoteOrchestrator, images_path: str) -> None:
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
        print(red("  Number out of range."))
        return None

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path)
        if img is None:
            print(red(f"  Could not read image: {path}"))
        return img

    # ── Modes ─────────────────────────────────────────────────────────────────

    def mode_full_flow(self) -> None:
        path = self._choose_image()
        if not path:
            return
        image = self._load_image(path)
        if image is None:
            return

        print(f"\n  {dim('Running full automated entry flow…')}")
        result, vision_output = self.orchestrator.run_automated_entry(image)
        print_result(result)

        display = vision_output["visual"] if vision_output else image
        if input("  Show annotated image? (y/n): ").strip().lower() == "y":
            show_image(display, window_title="Gate — Full Flow")

    def mode_vision_only(self) -> None:
        path = self._choose_image()
        if not path:
            return
        image = self._load_image(path)
        if image is None:
            return

        print(f"\n  {dim('Running vision pipeline…')}")
        vision_output = self.orchestrator.client.detect_plate(image)
        print_vision_result(vision_output)

        if vision_output:
            if input("  Show annotated image? (y/n): ").strip().lower() == "y":
                show_image(vision_output["visual"], window_title="Gate — Vision Only")

    def mode_audio_only(self) -> None:
        print(f"\n  {dim('Starting audio-only plate request…')}")
        audio_result = self.orchestrator._request_plate_from_driver(vision_output=None)
        spoken_plate = audio_result["plate"]
        print(f"\n  Parsed plate : {bold(spoken_plate)}")
        print(f"  Transcription: {dim(audio_result['transcription'])}")

        db_entry = self.orchestrator.client.lookup_plate(spoken_plate)
        if not db_entry:
            print(red(f"\n  Plate {spoken_plate!r} not found in database."))
            return

        print(green(f"\n  Plate found — {db_entry['driver_name']}, {db_entry['dock']}"))

        spoken_name = self.orchestrator._request_driver_name()
        print(f"\n  Driver said name: {bold(spoken_name)}")

        match = self.orchestrator.client.verify_name(db_entry["driver_name"], spoken_name)
        if match:
            dock_msg = (
                f"Access granted. Welcome, {db_entry['driver_name']}. "
                f"Please proceed to {db_entry['dock']}. "
                f"Your cargo is {db_entry['cargo']}. "
                f"Your arrival window is {db_entry['arrival_window']}. "
                "Have a safe unloading."
            )
            self.orchestrator._speak(dock_msg)
            print_result({"status": "success", "db_entry": db_entry})
        else:
            print_result({"status": "name_mismatch", "plate": db_entry["plate"]})

    def mode_batch(self) -> None:
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
            status   = result.get("status", "?")
            plate    = result.get("plate") or result.get("db_entry", {}).get("plate", "-")
            conf_str = f"{vision_output['conf']:.2f}" if vision_output else "n/a"
            rows.append((name, status, plate, conf_str))

        col_w = [
            max(len(r[i]) for r in rows + [("File", "Status", "Plate", "Conf")])
            for i in range(4)
        ]
        header = (
            f"  {'File':<{col_w[0]}}  {'Status':<{col_w[1]}}  "
            f"{'Plate':<{col_w[2]}}  {'Conf':<{col_w[3]}}"
        )
        print(bold(header))
        print("  " + "-" * (sum(col_w) + 8))
        for name, status, plate, conf in rows:
            colour = green if status == "success" else (
                     red   if status in ("alert_worker", "name_mismatch") else yellow)
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
    parser = argparse.ArgumentParser(description="Harbour Agent CLI test tool")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="Jetson API server URL, e.g. http://192.168.1.50:8000",
    )
    args = parser.parse_args()

    config = read_config("utils/config.yaml")

    client = HarbourClient(base_url=args.host)
    print(f"  Connecting to {args.host} …")
    try:
        health = client.health()
        print(f"  Server health: {health}\n")
    except Exception as exc:
        print(red(f"  Cannot reach server at {args.host}: {exc}"))
        raise SystemExit(1)

    orchestrator = RemoteOrchestrator(client=client)
    cli = CLI(orchestrator, config["images"]["path"])
    cli.run()
