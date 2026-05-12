"""
test_orchestrator.py — integration tests for Orchestrator.run_automated_entry().

Strategy
--------
VisionPipeline and AudioPipeline are patched at construction time so the
orchestrator uses MagicMock instances. The SQLite database is real (temp file)
and is seeded with create_mock_db() so lookup_plate() works as in production.

Flow map tested
---------------
Flow A — vision confident, plate in DB
  A1: name match          → success
  A2: name mismatch       → name_mismatch

Flow B — vision confident, plate NOT in DB
  B1: driver confirms YES → plate genuinely absent → alert_worker
  B2: driver says NO (bad read) → audio plate found → name match → success
  B3: driver says NO (bad read) → audio plate found → name mismatch
  B4: driver says NO (bad read) → audio plate also absent → alert_worker

Flow C — vision low-confidence or no detection
  C1: audio plate found   → name match → success
  C2: audio plate absent  → alert_worker
  C3: no detection at all → audio plate found → success
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from scripts.orchestrator import Orchestrator, create_mock_db


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_config(mock_config):
    """Config pointing at a real temp DB pre-populated with mock data."""
    create_mock_db(mock_config["database"]["db_path"])
    return mock_config


@pytest.fixture
def orch(seeded_config):
    """Orchestrator with VisionPipeline and AudioPipeline fully mocked."""
    with patch("scripts.orchestrator.VisionPipeline"), \
         patch("scripts.orchestrator.AudioPipeline"):
        o = Orchestrator(seeded_config, onnx=False)
    yield o
    o.close()


@pytest.fixture
def good_vision(mock_vision_output):
    """vision_output for a plate that IS in the mock DB (conf 0.92)."""
    return mock_vision_output          # plate="RK612AL", conf=0.92


@pytest.fixture
def bad_vision():
    """vision_output for a plate NOT in the mock DB (conf 0.91)."""
    return {
        "plate":  "XX999ZZ",
        "conf":   0.91,
        "bbox":   (10, 20, 100, 60),
        "visual": np.zeros((480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def low_conf_vision(mock_vision_output):
    """vision_output below the confidence threshold (0.35)."""
    out = dict(mock_vision_output)
    out["conf"] = 0.35
    return out


# ── Flow A: vision confident, plate in DB ────────────────────────────────────

class TestFlowA:

    def test_A1_success_on_name_match(self, orch, blank_image, good_vision):
        orch.vision_pipeline.read_plate.return_value = good_vision
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        result, vision_out = orch.run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert result["db_entry"]["plate"] == "RK612AL"
        assert result["db_entry"]["dock"] == "Dock B"
        assert vision_out is good_vision
        # Dock instructions must be spoken on success
        orch.audio_pipeline.give_dock_instructions.assert_called_once()

    def test_A1_vision_output_is_returned(self, orch, blank_image, good_vision):
        orch.vision_pipeline.read_plate.return_value = good_vision
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        _, vision_out = orch.run_automated_entry(blank_image)
        assert vision_out is good_vision

    def test_A2_name_mismatch_returns_correct_status(self, orch, blank_image, good_vision):
        orch.vision_pipeline.read_plate.return_value = good_vision
        orch.audio_pipeline.request_driver_name.return_value = "Wrong Person"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = False

        result, _ = orch.run_automated_entry(blank_image)

        assert result["status"] == "name_mismatch"
        assert result["plate"] == "RK612AL"

    def test_A2_alert_spoken_on_name_mismatch(self, orch, blank_image, good_vision):
        orch.vision_pipeline.read_plate.return_value = good_vision
        orch.audio_pipeline.request_driver_name.return_value = "Wrong Person"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = False

        orch.run_automated_entry(blank_image)
        orch.audio_pipeline.speak.assert_called()


# ── Flow B: vision confident, plate NOT in DB ─────────────────────────────────

class TestFlowB:

    def test_B1_confirmed_absent_plate_alerts_worker(self, orch, blank_image, bad_vision):
        orch.vision_pipeline.read_plate.return_value = bad_vision
        orch.audio_pipeline.confirm_plate_with_driver.return_value = True  # "yes that's correct"

        result, _ = orch.run_automated_entry(blank_image)

        assert result["status"] == "alert_worker"
        assert result["plate"] == "XX999ZZ"
        orch.audio_pipeline.speak.assert_called()

    def test_B1_plate_confirmed_so_no_audio_fallback_for_plate(self, orch, blank_image, bad_vision):
        orch.vision_pipeline.read_plate.return_value = bad_vision
        orch.audio_pipeline.confirm_plate_with_driver.return_value = True

        orch.run_automated_entry(blank_image)

        # Audio plate request should NOT have been called — driver confirmed reading
        orch.audio_pipeline.request_plate_from_driver.assert_not_called()

    def test_B2_denied_reading_then_audio_plate_found_name_match(
        self, orch, blank_image, bad_vision
    ):
        orch.vision_pipeline.read_plate.return_value = bad_vision
        orch.audio_pipeline.confirm_plate_with_driver.return_value = False   # "no that's wrong"
        orch.audio_pipeline.request_plate_from_driver.return_value = {
            "plate": "RK612AL",   # plate that IS in DB
            "transcription": "Romeo Kilo six one two alpha lima",
        }
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        result, _ = orch.run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert result["db_entry"]["driver_name"] == "Jane Doe"

    def test_B3_denied_reading_audio_plate_found_name_mismatch(
        self, orch, blank_image, bad_vision
    ):
        orch.vision_pipeline.read_plate.return_value = bad_vision
        orch.audio_pipeline.confirm_plate_with_driver.return_value = False
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
        orch.audio_pipeline.request_driver_name.return_value = "Imposter"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = False

        result, _ = orch.run_automated_entry(blank_image)
        assert result["status"] == "name_mismatch"

    def test_B4_denied_reading_audio_plate_also_absent(
        self, orch, blank_image, bad_vision
    ):
        orch.vision_pipeline.read_plate.return_value = bad_vision
        orch.audio_pipeline.confirm_plate_with_driver.return_value = False
        orch.audio_pipeline.request_plate_from_driver.return_value = {
            "plate": "YY000XX"    # also not in DB
        }

        result, _ = orch.run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"


# ── Flow C: low confidence or no detection ────────────────────────────────────

class TestFlowC:

    def test_C1_low_conf_audio_found_name_match(
        self, orch, blank_image, low_conf_vision
    ):
        orch.vision_pipeline.read_plate.return_value = low_conf_vision
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        result, _ = orch.run_automated_entry(blank_image)
        assert result["status"] == "success"

    def test_C1_confirm_is_not_called_on_low_confidence(
        self, orch, blank_image, low_conf_vision
    ):
        """confirm_plate_with_driver is only for high-conf reads not in DB."""
        orch.vision_pipeline.read_plate.return_value = low_conf_vision
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        orch.run_automated_entry(blank_image)
        orch.audio_pipeline.confirm_plate_with_driver.assert_not_called()

    def test_C2_low_conf_audio_plate_absent_alerts_worker(
        self, orch, blank_image, low_conf_vision
    ):
        orch.vision_pipeline.read_plate.return_value = low_conf_vision
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "YY000XX"}

        result, _ = orch.run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"

    def test_C3_no_detection_audio_found_success(self, orch, blank_image):
        orch.vision_pipeline.read_plate.return_value = None    # YOLO found nothing
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        result, vision_out = orch.run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert vision_out is None    # no visual to show

    def test_C3_no_detection_audio_absent_alerts_worker(self, orch, blank_image):
        orch.vision_pipeline.read_plate.return_value = None
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "YY000XX"}

        result, _ = orch.run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"


# ── Return type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    """The GUI unpacks (result, vision_output) — ensure this always holds."""

    @pytest.mark.parametrize("status", ["success", "alert_worker", "name_mismatch"])
    def test_always_returns_two_element_tuple(self, orch, blank_image, status):
        orch.vision_pipeline.read_plate.return_value = None
        if status == "success":
            orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
            orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
            orch.audio_pipeline.language_model.verify_name_similarity.return_value = True
        elif status == "name_mismatch":
            orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
            orch.audio_pipeline.request_driver_name.return_value = "Wrong"
            orch.audio_pipeline.language_model.verify_name_similarity.return_value = False
        else:  # alert_worker
            orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "YY000XX"}

        returned = orch.run_automated_entry(blank_image)

        assert isinstance(returned, tuple), "run_automated_entry must return a tuple"
        assert len(returned) == 2,          "tuple must have exactly two elements"
        result, _ = returned
        assert "status" in result,          "first element must have a 'status' key"

    def test_result_dict_has_plate_on_alert(self, orch, blank_image):
        orch.vision_pipeline.read_plate.return_value = None
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "YY000XX"}

        result, _ = orch.run_automated_entry(blank_image)
        assert "plate" in result

    def test_result_dict_has_db_entry_on_success(self, orch, blank_image):
        orch.vision_pipeline.read_plate.return_value = None
        orch.audio_pipeline.request_plate_from_driver.return_value = {"plate": "RK612AL"}
        orch.audio_pipeline.request_driver_name.return_value = "Jane Doe"
        orch.audio_pipeline.language_model.verify_name_similarity.return_value = True

        result, _ = orch.run_automated_entry(blank_image)
        assert "db_entry" in result
        assert result["db_entry"]["driver_name"] == "Jane Doe"


# ── lookup_plate ──────────────────────────────────────────────────────────────

class TestLookupPlate:

    def test_known_plate_returns_full_entry(self, orch):
        entry = orch.lookup_plate("RK612AL")
        assert entry is not None
        assert entry["driver_name"] == "Jane Doe"
        assert entry["cargo"] == "Chemicals"
        assert entry["dock"] == "Dock B"

    def test_unknown_plate_returns_none(self, orch):
        assert orch.lookup_plate("XXXXXX") is None

    def test_plate_lookup_is_case_sensitive(self, orch):
        """DB stores plates in the exact case from create_mock_db()."""
        assert orch.lookup_plate("rk612al") is None   # lowercase → not found
