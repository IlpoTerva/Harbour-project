"""
conftest.py — shared mocks and fixtures for the harbour gate test suite.

Mocking strategy
----------------
Heavy ML dependencies (YOLO, Whisper, Piper, Llama, sounddevice) are injected
into sys.modules *before* any test file imports scripts.*. This means:
  - The tests never need model files on disk.
  - All ML calls are MagicMocks by default; individual tests configure
    return values to exercise specific code paths.

Run from the project root:
    pytest tests/ -v
"""

import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# ── Prevent ModuleNotFoundError for ML / hardware dependencies ────────────────
for _mod in [
    "llama_cpp",
    "faster_whisper",
    "piper",
    "sounddevice",
    "ultralytics",
    "fast_plate_ocr",
]:
    sys.modules[_mod] = MagicMock()


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_config(tmp_path):
    """Minimal config dict. DB is a real temp file so SQLite works normally."""
    return {
        "models": {
            "model_path":         "fake_model.pt",
            "model_path_onnx":    "fake_model.onnx",
            "whisper_model_path": "small",
            "piper_model_path":   "fake_voice.onnx",
            "llm_model_path":     "fake_llm.gguf",
        },
        "database": {"db_path": str(tmp_path / "test.db")},
        "images":   {"path":    str(tmp_path)},
    }


@pytest.fixture
def blank_image():
    """480×640 black BGR image — the minimum a real camera would produce."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_vision_output():
    """Realistic return value from VisionPipeline.read_plate() on success."""
    return {
        "plate":  "RK612AL",
        "conf":   0.92,
        "bbox":   (10, 20, 100, 60),
        "visual": np.zeros((480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def mock_db_entry():
    """DB row dict matching one of the rows in create_mock_db()."""
    return {
        "plate":          "RK612AL",
        "driver_name":    "Jane Doe",
        "cargo":          "Chemicals",
        "dock":           "Dock B",
        "arrival_window": "09:00-11:00",
    }


def make_yolo_box(x1=10, y1=20, x2=100, y2=60, conf=0.9):
    """Return a MagicMock that mimics a single YOLO bounding box."""
    box = MagicMock()
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=float)]
    box.conf  = [conf]
    return box


def make_yolo_results(boxes, plot_image=None):
    """Return a list that mimics the list YOLO() returns.

    boxes      – list of MagicMock boxes from make_yolo_box()
    plot_image – optional annotated BGR array; defaults to a black image
    """
    if plot_image is None:
        plot_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = MagicMock()
    result.boxes = boxes
    result.plot.return_value = plot_image
    return [result]


def make_ocr_prediction(plate="RK612AL", char_conf=0.9):
    """Return a MagicMock mimicking a fast-plate-ocr Prediction object."""
    pred = MagicMock()
    pred.plate      = plate
    pred.char_probs = np.full(len(plate), char_conf)
    return pred


def make_llm_response(text: str) -> dict:
    """Return a dict shaped like a llama-cpp-python response."""
    return {"choices": [{"text": f" {text}\n"}]}
