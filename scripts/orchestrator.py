import logging
import sqlite3
import numpy as np
from typing import Dict, Any, Optional, Tuple

from scripts.vision_pipeline import VisionPipeline, read_config
from scripts.audio_pipeline import AudioPipeline
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


# ── Database setup ────────────────────────────────────────────────────────────

def create_mock_db(db_path: str = "license_plates.db") -> None:
    """Create and populate the demo database.

    Schema includes cargo manifest, dock assignment, and expected arrival window
    so the gate agent has something meaningful to communicate to the driver.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id              INTEGER PRIMARY KEY,
            plate           TEXT    NOT NULL UNIQUE,
            cargo           TEXT,
            dock            TEXT,
            arrival_window  TEXT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    mock_data = [
        ("PP587A0", "Electronics",      "Dock A", "08:00-10:00"),
        ("RK612AL", "Chemicals",        "Dock B", "09:00-11:00"),
        ("LM633BD", "Machinery",        "Dock C", "10:00-12:00"),
        ("RK763AS", "Food",             "Dock A", "11:00-13:00"),
        ("BZM2227", "Pharmaceuticals",  "Dock D", "12:00-14:00"),
        ("RK603AV", "Construction",     "Dock B", "13:00-15:00"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO plates (plate, cargo, dock, arrival_window) VALUES (?, ?, ?, ?)",
        mock_data,
    )
    conn.commit()
    conn.close()


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Coordinates VisionPipeline, AudioPipeline, and the SQLite database.

    Use as a context manager to ensure the DB connection is closed cleanly:

        with Orchestrator() as orch:
            result, vision_output = orch.read_plate(image)
    """

    def __init__(self, config: Dict[str, Any],onnx=False) -> None:
        self.vision_pipeline = VisionPipeline(config=config, onnx=onnx)
        self.audio_pipeline = AudioPipeline(conf=config)
        # check_same_thread=False is required because the audio fallback runs
        # in a background thread and also calls lookup_plate().
        self.db_connection = sqlite3.connect(config["database"]["db_path"], check_same_thread=False)

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

    # ── Database helpers ──────────────────────────────────────────────────────

    def lookup_plate(self, plate_text: str) -> Optional[Dict[str, Any]]:
        """Return the manifest row for `plate_text`, or None if not found."""
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT plate, cargo, dock, arrival_window FROM plates WHERE plate = ?",
            (plate_text,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "plate":          row[0],
            "cargo":          row[1],
            "dock":           row[2],
            "arrival_window": row[3],
        }

    # ── Vision path ───────────────────────────────────────────────────────────

    def read_plate(self, image: np.ndarray) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Run the vision pipeline on `image` and look up the result in the DB.

        Always returns a (result, vision_output) pair so the GUI can display the
        annotated image regardless of confidence or DB status.

        result["status"] values
        -----------------------
        "recognized"      plate read with confidence >= 0.5 and found in DB
        "not_in_db"       plate read with confidence >= 0.5 but absent from DB
        "low_confidence"  plate detected but OCR confidence < 0.5
        "no_detection"    YOLO found no plate at all
        """
        vision_output = self.vision_pipeline.read_plate(image)

        if vision_output is None:
            return {"status": "no_detection"}, None

        conf = vision_output["conf"]
        plate_text = vision_output["plate"]

        if conf < 0.5:
            logger.info(f"Low OCR confidence ({conf:.2f}) for {plate_text!r} — audio fallback needed.")
            return {"status": "low_confidence", "plate": plate_text, "conf": conf}, vision_output

        db_entry = self.lookup_plate(plate_text)
        if db_entry:
            logger.info(f"Plate {plate_text!r} recognised — {db_entry['dock']}, cargo: {db_entry['cargo']}")
            return {
                "status":   "recognized",
                "plate":    plate_text,
                "conf":     conf,
                "db_entry": db_entry,
            }, vision_output

        logger.info(f"Plate {plate_text!r} not found in database.")
        return {"status": "not_in_db", "plate": plate_text, "conf": conf}, vision_output

    # ── Audio fallback path ───────────────────────────────────────────────────

    def handle_audio_fallback(
        self, vision_output: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Speak a prompt, listen for the driver's plate, and look it up.

        Intended to run in a background thread from the GUI.

        Returns:
            (db_entry_or_None, spoken_plate_text)
        """
        audio_result = self.audio_pipeline.request_plate_from_driver(vision_output)
        spoken_plate = audio_result["plate"]
        db_entry = self.lookup_plate(spoken_plate)

        if db_entry:
            logger.info(f"Driver said {spoken_plate!r} — found in manifest.")
            self.audio_pipeline.driver_instructions(inDatabase=True)
        else:
            logger.info(f"Driver said {spoken_plate!r} — NOT found in manifest.")
            self.audio_pipeline.driver_instructions(inDatabase=False, read_plate=spoken_plate)
        return db_entry, spoken_plate
