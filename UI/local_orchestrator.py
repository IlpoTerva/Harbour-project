"""
Local orchestrator for the Harbour Agent — runs entirely on the Jetson.

Mirrors RemoteOrchestrator's public interface but calls ML pipeline objects
directly instead of delegating to HarbourClient over HTTP.  Audio I/O
(recording + playback) uses sounddevice against the Jetson's USB audio device.

Usage:
    from local_orchestrator import LocalOrchestrator
    orch = LocalOrchestrator(config=read_config("utils/config.yaml"))
    result, vision_output = orch.run_automated_entry(image)
"""

import collections
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audio_pipeline import LanguageModel, Listener, Speaker
from scripts.helpers import create_mock_db
from scripts.vision_pipeline import VisionPipeline
from i18n import get as _t, SUPPORTED_LANGS

_SAMPLE_RATE         = 16_000
_CHANNELS            = 1
_VAD_FRAME_SAMPLES   = 480   # 30 ms per frame at 16 kHz
_ENERGY_THRESHOLD    = 0.01  # RMS amplitude; raise to 0.02–0.03 in noisy environments
_SPEECH_ONSET_FRAMES = 3     # ~90 ms of speech required to start recording
_SILENCE_STOP_FRAMES = 20    # ~600 ms of silence after speech to stop recording
_PRE_SPEECH_FRAMES   = 10    # ~300 ms pre-roll so speech onset is not clipped
_VAD_MAX_DURATION    = 15.0  # hard cap in seconds

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class LocalOrchestrator:
    """
    Full gate-check pipeline running locally on the Jetson — no HTTP layer.

    All ML inference and DB lookups are handled directly by the pipeline
    objects.  Language auto-detects from the first driver transcription and
    resets to default_language at the start of each new session.

    Public interface is identical to RemoteOrchestrator:
        result, vision_output = orchestrator.run_automated_entry(image)
    """

    def __init__(self, config: Dict[str, Any], default_language: str = "en") -> None:
        self._conf = config
        self._default_lang = default_language
        self._lang = default_language
        self.on_vision_result = None  # optional callback: fn(vision_output)

        device = self._get_device()
        logger.info(f"LocalOrchestrator using device: {device}")

        try:
            self._vision = VisionPipeline(config=config, device=device, onnx=True)
            logger.info("VisionPipeline loaded in ONNX mode.")
        except Exception as exc:
            logger.warning(f"ONNX init failed ({exc}), falling back to PyTorch mode.")
            self._vision = VisionPipeline(config=config, device=device, onnx=False)

        self._listener = Listener(conf=config, device=device)
        self._speaker  = Speaker(conf=config, device=device)
        self._llm      = LanguageModel(conf=config, device=device)

        db_path = config["database"]["db_path"]
        if not os.path.exists(db_path):
            create_mock_db(db_path)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        logger.info("All models loaded — ready.")

    @staticmethod
    def _get_device() -> str:
        try:
            import onnxruntime as ort
            return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        except ImportError:
            return "cpu"

    # ── Audio I/O ─────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        logger.info(f"Speaking [{self._lang}]: {text!r}")
        self._speaker.speak(text, language=self._lang)

    def _record_vad(self) -> np.ndarray:
        """Record until speech ends (VAD), with a hard cap of _VAD_MAX_DURATION."""
        logger.info("Listening (VAD)…")
        frames: list = []
        pre_buffer: collections.deque = collections.deque(maxlen=_PRE_SPEECH_FRAMES)
        speech_onset = 0
        silence_count = 0
        speech_started = False
        max_frames = int(_VAD_MAX_DURATION * _SAMPLE_RATE / _VAD_FRAME_SAMPLES)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
            blocksize=_VAD_FRAME_SAMPLES,
        ) as stream:
            for _ in range(max_frames):
                block, _ = stream.read(_VAD_FRAME_SAMPLES)
                block = block.flatten()
                rms = float(np.sqrt(np.mean(block ** 2)))

                if not speech_started:
                    pre_buffer.append(block)
                    if rms >= _ENERGY_THRESHOLD:
                        speech_onset += 1
                        if speech_onset >= _SPEECH_ONSET_FRAMES:
                            speech_started = True
                            frames.extend(pre_buffer)
                            pre_buffer.clear()
                            silence_count = 0
                    else:
                        speech_onset = 0
                else:
                    frames.append(block)
                    if rms < _ENERGY_THRESHOLD:
                        silence_count += 1
                        if silence_count >= _SILENCE_STOP_FRAMES:
                            break
                    else:
                        silence_count = 0

        if not frames:
            logger.warning("VAD: no speech detected within timeout")
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(frames)
        logger.info(f"VAD: captured {len(audio) / _SAMPLE_RATE:.2f}s of audio")
        return audio

    def _listen_and_transcribe(self) -> Dict[str, Any]:
        audio = self._record_vad()
        text, detected_lang = self._listener.transcribe(audio)
        if detected_lang and detected_lang in SUPPORTED_LANGS:
            if detected_lang != self._lang:
                logger.info(f"Language switched: {self._lang!r} → {detected_lang!r}")
            self._lang = detected_lang
        return {"transcription": text, "audio": audio}

    # ── Conversation helpers ──────────────────────────────────────────────────

    def _request_plate_from_driver(
        self, vision_output: Optional[Dict]
    ) -> Dict[str, Any]:
        if vision_output and vision_output.get("conf", 0) >= 0.5:
            plate_letters = " ".join(vision_output["plate"])
            prompt = _t("plate_low_conf", self._lang, plate=plate_letters)
        else:
            prompt = _t("plate_not_read", self._lang)
        self._speak(prompt)
        result = self._listen_and_transcribe()
        result["plate"] = self._llm.parse_plate_from_transcription(result["transcription"])
        logger.info(f"Driver-provided plate: {result['plate']!r}")
        return result

    def _confirm_plate_with_driver(self, plate: str) -> bool:
        plate_letters = " ".join(plate)
        self._speak(_t("confirm_plate", self._lang, plate=plate_letters))
        result = self._listen_and_transcribe()
        confirmed = self._llm.parse_yes_no(result["transcription"])
        logger.info(f"Plate {plate!r} confirmed: {confirmed}")
        return confirmed

    def _request_driver_name(self) -> str:
        self._speak(_t("request_name", self._lang))
        result = self._listen_and_transcribe()
        name = self._llm.extract_name(result["transcription"])
        logger.info(f"Driver name: {name!r}")
        return name

    # ── Flow helpers ──────────────────────────────────────────────────────────

    def _alert(
        self, reason: str, plate: Optional[str], vision_output: Any
    ) -> Tuple[Dict, Any]:
        if reason == "plate_not_in_db":
            plate_str = " ".join(plate) if plate else "unknown"
            msg = _t("alert_not_in_db", self._lang, plate=plate_str)
        elif reason == "name_mismatch":
            msg = _t("alert_name_mismatch", self._lang)
        else:
            msg = _t("alert_generic", self._lang)

        self._speak(msg)
        logger.warning(f"WORKER ALERT — reason: {reason} | plate: {plate!r}")
        status = "alert_worker" if reason == "plate_not_in_db" else reason
        return {"status": status, "plate": plate}, vision_output

    def _verify_name(
        self, db_entry: Dict, vision_output: Any
    ) -> Tuple[Dict, Any]:
        spoken_name = self._request_driver_name()
        name_ok = self._llm.verify_name_similarity(db_entry["driver_name"], spoken_name)

        if name_ok:
            logger.info(f"Access granted — plate: {db_entry['plate']!r}, driver: {spoken_name!r}")
            dock_msg = _t(
                "access_granted", self._lang,
                name=db_entry["driver_name"],
                dock=db_entry["dock"],
                cargo=db_entry["cargo"],
                window=db_entry["arrival_window"],
            )
            self._speak(dock_msg)
            return {"status": "success", "db_entry": db_entry}, vision_output
        else:
            logger.warning(
                f"Name mismatch — expected: {db_entry['driver_name']!r}, got: {spoken_name!r}"
            )
            return self._alert("name_mismatch", db_entry["plate"], vision_output)

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _lookup_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT plate, driver_name, cargo, dock, arrival_window "
            "FROM license_plates WHERE plate = ?",
            (plate,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "plate":          row[0],
            "driver_name":    row[1],
            "cargo":          row[2],
            "dock":           row[3],
            "arrival_window": row[4],
        }

    def list_all_plates(self) -> list:
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT plate, driver_name, cargo, dock, arrival_window "
            "FROM license_plates ORDER BY id"
        )
        rows = cursor.fetchall()
        return [
            {
                "plate":          row[0],
                "driver_name":    row[1],
                "cargo":          row[2],
                "dock":           row[3],
                "arrival_window": row[4],
            }
            for row in rows
        ]

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_automated_entry(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Full gate-check cycle.  Language resets to the configured default at the
        start of each session and switches to the driver's detected language after
        the first transcription.

        Flow A — Vision confident (conf ≥ 0.5) + plate in DB:
            vision ──► DB hit ──► name verify ──► grant / deny

        Flow B — Vision confident + plate NOT in DB:
            vision ──► DB miss ──► confirm reading with driver
                ├─ driver confirms ──► alert worker
                └─ driver denies   ──► ask plate via audio ──► name verify / alert

        Flow C — Vision low-confidence or no detection:
            audio plate request ──► DB lookup ──► name verify / alert
        """
        self._lang = self._default_lang

        vision_output = self._vision.read_plate(image)
        if self.on_vision_result is not None:
            self.on_vision_result(vision_output)

        db_entry = None
        plate_text = None

        if vision_output and vision_output["conf"] >= 0.5:
            plate_text = vision_output["plate"]
            db_entry = self._lookup_plate(plate_text)

            if db_entry is None:
                logger.info(f"Plate {plate_text!r} not in DB. Asking driver to confirm.")
                if self._confirm_plate_with_driver(plate_text):
                    return self._alert("plate_not_in_db", plate_text, vision_output)
                audio_result = self._request_plate_from_driver(vision_output)
                plate_text = audio_result["plate"]
                db_entry = self._lookup_plate(plate_text)
        else:
            logger.info("Vision insufficient. Requesting plate verbally.")
            audio_result = self._request_plate_from_driver(vision_output)
            plate_text = audio_result["plate"]
            db_entry = self._lookup_plate(plate_text)

        if db_entry is None:
            return self._alert("plate_not_in_db", plate_text, vision_output)

        return self._verify_name(db_entry, vision_output)

    def close(self) -> None:
        self._db.close()
