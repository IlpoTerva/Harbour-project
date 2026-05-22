import sounddevice as sd
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple
from client import HarbourClient

_SAMPLE_RATE = 16_000


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
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