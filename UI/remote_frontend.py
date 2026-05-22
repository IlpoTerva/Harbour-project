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
import base64
import io
import logging
import os
import threading
import wave
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import sounddevice as sd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000  # Hz — must match the server's Whisper input expectation


# ── Audio codec helpers ───────────────────────────────────────────────────────

def _float32_to_wav(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Encode a float32 numpy array as a mono WAV byte string for upload."""
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_to_float32(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes (from the TTS endpoint) to (float32 array, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


# ── HTTP client ───────────────────────────────────────────────────────────────

class HarbourClient:
    """
    Thin wrapper around the Jetson API endpoints defined in scripts/api_server.py.

    All network calls are synchronous; the caller is responsible for running
    them off the main thread when needed (RemoteOrchestrator is always called
    from RemoteGUI's background worker thread).
    """

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base}/health", timeout=10).json()

    def detect_plate(self, image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Upload a BGR image and return the detection result, or None."""
        _, jpg = cv2.imencode(".jpg", image_bgr)
        resp = requests.post(
            f"{self.base}/vision/detect",
            files={"image": ("image.jpg", jpg.tobytes(), "image/jpeg")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        jpg_bytes = base64.b64decode(data["visual_b64"])
        arr = np.frombuffer(jpg_bytes, np.uint8)
        visual = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return {
            "plate":  data["plate"],
            "conf":   data["conf"],
            "bbox":   tuple(data["bbox"]),
            "visual": visual,
        }

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Request TTS synthesis. Returns (float32 audio array, sample_rate)."""
        resp = requests.post(
            f"{self.base}/tts/synthesize",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _wav_to_float32(resp.content)

    def transcribe(self, audio: np.ndarray) -> str:
        """Upload recorded audio and return the Whisper transcription."""
        wav_bytes = _float32_to_wav(audio)
        resp = requests.post(
            f"{self.base}/stt/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def parse_plate(self, transcription: str) -> str:
        resp = requests.post(
            f"{self.base}/llm/parse_plate",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["plate"]

    def parse_yes_no(self, transcription: str) -> bool:
        resp = requests.post(
            f"{self.base}/llm/parse_yes_no",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def extract_name(self, transcription: str) -> str:
        resp = requests.post(
            f"{self.base}/llm/extract_name",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["name"]

    def verify_name(self, db_name: str, spoken_name: str) -> bool:
        resp = requests.post(
            f"{self.base}/llm/verify_name",
            json={"db_name": db_name, "spoken_name": spoken_name},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["match"]

    def lookup_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        resp = requests.post(
            f"{self.base}/db/lookup",
            json={"plate": plate},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["entry"] if data["found"] else None


# ── Remote Orchestrator ───────────────────────────────────────────────────────

class RemoteOrchestrator:
    """
    Mirrors scripts.orchestrator.Orchestrator but delegates all ML and DB work
    to the Jetson server via HarbourClient.  Audio I/O (recording + playback)
    is handled locally on the laptop with sounddevice.

    The same public interface as Orchestrator:
        result, vision_output = remote_orchestrator.run_automated_entry(image)
    """

    def __init__(self, client: HarbourClient) -> None:
        self.client = client
        self.on_vision_result = None  # optional callback: fn(vision_output) called right after detection

    # ── Local audio I/O ───────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        """Synthesise on the Jetson and play back through the laptop speaker."""
        logger.info(f"Speaking: {text!r}")
        audio, sample_rate = self.client.synthesize(text)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def _record(self, duration: int) -> np.ndarray:
        logger.info(f"Recording for {duration}s…")
        audio = sd.rec(
            int(duration * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def _listen_and_transcribe(self, duration: int) -> Dict[str, Any]:
        audio = self._record(duration)
        text = self.client.transcribe(audio)
        return {"transcription": text, "audio": audio}

    # ── Conversation helpers (mirrors AudioPipeline) ──────────────────────────

    def _request_plate_from_driver(
        self, vision_output: Optional[Dict], duration: int = 6
    ) -> Dict[str, Any]:
        if vision_output and vision_output.get("conf", 0) >= 0.5:
            plate_letters = " ".join(vision_output["plate"])
            prompt = (
                f"I detected the plate {plate_letters} but I am not fully confident. "
                "Could you please confirm your licence plate out loud?"
            )
        else:
            prompt = "I could not read your licence plate. Please say it out loud now."
        self._speak(prompt)
        result = self._listen_and_transcribe(duration)
        result["plate"] = self.client.parse_plate(result["transcription"])
        logger.info(f"Driver-provided plate: {result['plate']!r}")
        return result

    def _confirm_plate_with_driver(self, plate: str, duration: int = 4) -> bool:
        plate_letters = " ".join(plate)
        self._speak(
            f"I read your licence plate as {plate_letters}. "
            "Is that correct? Please say yes or no."
        )
        result = self._listen_and_transcribe(duration)
        confirmed = self.client.parse_yes_no(result["transcription"])
        logger.info(f"Plate {plate!r} confirmed: {confirmed}")
        return confirmed

    def _request_driver_name(self, duration: int = 5) -> str:
        self._speak("Please say your full name for verification.")
        result = self._listen_and_transcribe(duration)
        name = self.client.extract_name(result["transcription"])
        logger.info(f"Driver name: {name!r}")
        return name

    # ── Flow helpers (mirrors Orchestrator._alert / _verify_name) ────────────

    def _alert(
        self, reason: str, plate: Optional[str], vision_output: Any
    ) -> Tuple[Dict, Any]:
        if reason == "plate_not_in_db":
            plate_str = " ".join(plate) if plate else "unknown"
            msg = (
                f"I'm sorry, but the plate {plate_str} is not registered for today. "
                "A gate worker has been alerted. Please wait."
            )
        elif reason == "name_mismatch":
            msg = (
                "I'm sorry, but the name you provided does not match our records. "
                "A gate worker has been alerted. Please wait."
            )
        else:
            msg = "Access denied. A gate worker has been alerted. Please wait."

        self._speak(msg)
        logger.warning(f"WORKER ALERT — reason: {reason} | plate: {plate!r}")
        status = "alert_worker" if reason == "plate_not_in_db" else reason
        return {"status": status, "plate": plate}, vision_output

    def _verify_name(
        self, db_entry: Dict, vision_output: Any
    ) -> Tuple[Dict, Any]:
        spoken_name = self._request_driver_name()
        name_ok = self.client.verify_name(db_entry["driver_name"], spoken_name)

        if name_ok:
            logger.info(f"Access granted — plate: {db_entry['plate']!r}, driver: {spoken_name!r}")
            dock_msg = (
                f"Access granted. Welcome, {db_entry['driver_name']}. "
                f"Please proceed to {db_entry['dock']}. "
                f"Your cargo is {db_entry['cargo']}. "
                f"Your arrival window is {db_entry['arrival_window']}. "
                "Have a safe unloading."
            )
            self._speak(dock_msg)
            return {"status": "success", "db_entry": db_entry}, vision_output
        else:
            logger.warning(
                f"Name mismatch — expected: {db_entry['driver_name']!r}, got: {spoken_name!r}"
            )
            return self._alert("name_mismatch", db_entry["plate"], vision_output)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_automated_entry(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Full gate-check cycle.  Same interface as Orchestrator.run_automated_entry
        so RemoteGUI (and the existing GUI class) can use it without modification.

        Flow A — Vision confident (conf ≥ 0.5) + plate in DB:
            vision ──► DB hit ──► name verify ──► grant / deny

        Flow B — Vision confident + plate NOT in DB:
            vision ──► DB miss ──► confirm reading with driver
                ├─ driver confirms ──► alert worker
                └─ driver denies   ──► ask plate via audio ──► name verify / alert

        Flow C — Vision low-confidence or no detection:
            audio plate request ──► DB lookup ──► name verify / alert
        """
        # Phase 1: vision (on Jetson)
        vision_output = self.client.detect_plate(image)
        if self.on_vision_result is not None:
            self.on_vision_result(vision_output)

        db_entry = None
        plate_text = None

        if vision_output and vision_output["conf"] >= 0.5:
            plate_text = vision_output["plate"]
            db_entry = self.client.lookup_plate(plate_text)

            if db_entry is None:
                # Flow B
                logger.info(f"Plate {plate_text!r} not in DB. Asking driver to confirm.")
                if self._confirm_plate_with_driver(plate_text):
                    return self._alert("plate_not_in_db", plate_text, vision_output)
                audio_result = self._request_plate_from_driver(vision_output)
                plate_text = audio_result["plate"]
                db_entry = self.client.lookup_plate(plate_text)
        else:
            # Flow C
            logger.info("Vision insufficient. Requesting plate verbally.")
            audio_result = self._request_plate_from_driver(vision_output)
            plate_text = audio_result["plate"]
            db_entry = self.client.lookup_plate(plate_text)

        if db_entry is None:
            return self._alert("plate_not_in_db", plate_text, vision_output)

        # Phase 3: name verification
        return self._verify_name(db_entry, vision_output)


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
