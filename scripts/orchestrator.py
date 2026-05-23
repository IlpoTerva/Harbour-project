import logging
import sqlite3
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple

from scripts.vision_pipeline import VisionPipeline, read_config
from scripts.audio_pipeline import AudioPipeline

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Database setup ────────────────────────────────────────────────────────────

def create_mock_db(db_path: str = "license_plates_database.db") -> None:
    """Create and populate the demo database with driver names, cargo, dock, and window."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS license_plates (
            id              INTEGER PRIMARY KEY,
            plate           TEXT    NOT NULL UNIQUE,
            driver_name     TEXT,
            cargo           TEXT,
            dock            TEXT,
            arrival_window  TEXT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    mock_data = [
        ("PP587A0", "John Smith",        "Electronics",     "Dock A", "08:00-10:00"),
        ("RK612AL", "Jane Doe",          "Chemicals",       "Dock B", "09:00-11:00"),
        ("LM633BD", "Matti Meikalainen", "Machinery",       "Dock C", "10:00-12:00"),
        ("BZM2227", "Carlos Soler",      "Pharmaceuticals", "Dock D", "12:00-14:00"),
        ("RK603AV", "Alice Johnson",     "Construction",    "Dock B", "13:00-15:00"),
        ("RK763AS", "Bob Wilson",        "Food",            "Dock A", "11:00-13:00"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO license_plates "
        "(plate, driver_name, cargo, dock, arrival_window) VALUES (?, ?, ?, ?, ?)",
        mock_data,
    )
    conn.commit()
    conn.close()


# ── Orchestrator ──────────────────────────────────────────────────────────────
"""
Wrapper that coordinates VisionPipeline, AudioPipeline, and SQLite for the full vehicle entry flow.
NOTE - this is the public interface used by the GUI. All methods here are called by the GUI;
Might not be used since now used the api server instead, but keeping for reference and potential future use if we want to switch back to local inference.
"""
class Orchestrator:
    """
    Coordinates VisionPipeline, AudioPipeline, and SQLite.

    Single public entry point:
        result, vision_output = orchestrator.run_automated_entry(image)

    result["status"] values
    -----------------------
    "success"        - plate and name verified, dock instructions given
    "alert_worker"   - plate not found in DB after all fallbacks
    "name_mismatch"  - plate found but driver name does not match
    """

    def __init__(self, config: Dict[str, Any], onnx: bool = False) -> None:
        self.vision_pipeline = VisionPipeline(config=config, onnx=onnx)
        self.audio_pipeline = AudioPipeline(conf=config)
        # check_same_thread=False: the background GUI thread also calls lookup_plate()
        self.db_connection = sqlite3.connect(
            config["database"]["db_path"], check_same_thread=False
        )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        try:
            self.db_connection.close()
        except Exception:
            pass

    # ── Database helper ───────────────────────────────────────────────────────

    def lookup_plate(self, plate_text: str) -> Optional[Dict[str, Any]]:
        """Return the full manifest row for `plate_text`, or None if absent."""
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT plate, driver_name, cargo, dock, arrival_window "
            "FROM license_plates WHERE plate = ?",
            (plate_text,),
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _alert(self, reason: str, plate: Optional[str], vision_output) -> Tuple[Dict, Optional[Dict]]:
        """Speak a denial message, log a worker alert, and return the result tuple."""
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

        self.audio_pipeline.speak(msg)
        logger.warning(f"WORKER ALERT — reason: {reason} | plate: {plate!r}")
        print(f"\n🚨 WORKER ALERT — {reason.upper()} | plate: {plate}\n")

        status = "alert_worker" if reason == "plate_not_in_db" else reason
        return {"status": status, "plate": plate}, vision_output

    def _verify_name(
        self,
        db_entry: Dict[str, Any],
        vision_output: Optional[Dict[str, Any]],
    ) -> Tuple[Dict, Optional[Dict]]:
        """Ask for the driver's name, verify it, and grant or deny access."""
        spoken_name = self.audio_pipeline.request_driver_name()

        name_ok = self.audio_pipeline.language_model.verify_name_similarity(
            db_name=db_entry["driver_name"],
            spoken_name=spoken_name,
        )

        if name_ok:
            logger.info(
                f"Access granted — plate: {db_entry['plate']!r}, "
                f"driver: {spoken_name!r}"
            )
            self.audio_pipeline.give_dock_instructions(db_entry)
            return {"status": "success", "db_entry": db_entry}, vision_output
        else:
            logger.warning(
                f"Name mismatch — expected: {db_entry['driver_name']!r}, "
                f"got: {spoken_name!r}"
            )
            return self._alert("name_mismatch", db_entry["plate"], vision_output)

    # ── Main entry flow ───────────────────────────────────────────────────────

    def run_automated_entry(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Run the full vehicle entry flow and always return (result, vision_output).

        Flow A — Vision confident, plate in DB:
            vision ──► DB hit ──► name verify ──► grant / deny

        Flow B — Vision confident, plate NOT in DB:
            vision ──► DB miss ──► confirm reading with driver
                ├─ driver says YES (reading was correct) ──► alert worker
                └─ driver says NO  (reading was wrong)   ──► ask plate via audio
                        ├─ DB hit ──► name verify ──► grant / deny
                        └─ DB miss ──► alert worker

        Flow C — Vision low-confidence or no detection:
            vision ──► audio fallback for plate
                ├─ DB hit ──► name verify ──► grant / deny
                └─ DB miss ──► alert worker
        """

        # ── Phase 1: vision ───────────────────────────────────────────────────
        vision_output = self.vision_pipeline.read_plate(image)

        # ── Phase 2: determine db_entry ───────────────────────────────────────
        db_entry = None
        plate_text = None

        if vision_output and vision_output["conf"] >= 0.5:
            # Vision read a plate confidently
            plate_text = vision_output["plate"]
            db_entry = self.lookup_plate(plate_text)

            if db_entry is None:
                # ── Flow B: confident read but not in DB ──────────────────────
                logger.info(
                    f"Plate {plate_text!r} not in DB. "
                    "Asking driver to confirm the reading."
                )
                confirmed = self.audio_pipeline.confirm_plate_with_driver(plate_text)

                if confirmed:
                    # Driver agrees — the reading was correct but it's still not in DB
                    logger.warning(
                        f"Driver confirmed {plate_text!r} but it is not registered."
                    )
                    return self._alert("plate_not_in_db", plate_text, vision_output)
                else:
                    # Driver says the reading was wrong — ask them to say it
                    logger.info("Driver denied reading. Requesting plate verbally.")
                    audio_result = self.audio_pipeline.request_plate_from_driver(vision_output)
                    plate_text = audio_result["plate"]
                    db_entry = self.lookup_plate(plate_text)

        else:
            # ── Flow C: low confidence or no detection ────────────────────────
            logger.info("Vision insufficient. Requesting plate verbally.")
            audio_result = self.audio_pipeline.request_plate_from_driver(vision_output)
            plate_text = audio_result["plate"]
            db_entry = self.lookup_plate(plate_text)

        # If still no DB entry after all attempts, alert
        if db_entry is None:
            logger.warning(f"Plate {plate_text!r} not found after all attempts.")
            return self._alert("plate_not_in_db", plate_text, vision_output)

        # ── Phase 3: name verification ────────────────────────────────────────
        return self._verify_name(db_entry, vision_output)
