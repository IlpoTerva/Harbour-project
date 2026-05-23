import logging
import sqlite3
from typing import Dict, Any
import yaml

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

def read_config(path: str) -> Dict[str, Any]:
    """Read the YAML config file at `path` and return it as a dict."""
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

