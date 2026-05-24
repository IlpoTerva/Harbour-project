"""
test_orchestrator.py — integration tests for RemoteOrchestrator.run_automated_entry().

Strategy
--------
HarbourClient is replaced with a MagicMock so no HTTP calls are made.
sounddevice is already mocked by conftest.py, so no audio hardware is needed.
No real database is required — all DB access goes through client.lookup_plate().

Flow map tested
---------------
Flow A — vision confident (conf ≥ 0.5), plate in DB
  A1: name match          → success
  A2: name mismatch       → name_mismatch

Flow B — vision confident, plate NOT in DB
  B1: driver confirms YES → plate genuinely absent → alert_worker
  B2: driver says NO      → audio plate found → name match → success
  B3: driver says NO      → audio plate found → name mismatch
  B4: driver says NO      → audio plate also absent → alert_worker

Flow C — vision low-confidence or no detection
  C1: audio plate found   → name match → success
  C2: audio plate absent  → alert_worker
  C3: no detection (None) → audio plate found → success
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from remote_orchestrator import RemoteOrchestrator


# ── Shared test data ──────────────────────────────────────────────────────────

_DB_ENTRY = {
    "plate":          "RK612AL",
    "driver_name":    "Jane Doe",
    "cargo":          "Chemicals",
    "dock":           "Dock B",
    "arrival_window": "09:00-11:00",
}


# ── Client factory ────────────────────────────────────────────────────────────

def _client(**overrides) -> MagicMock:
    """Return a HarbourClient mock with audio defaults pre-set.

    Pass keyword args to override specific return values, e.g.:
        _client(detect_plate=vision_output, lookup_plate=_DB_ENTRY)
    """
    c = MagicMock()
    c.synthesize.return_value = (np.zeros(100, dtype=np.float32), 16_000)
    c.transcribe.return_value = ("", "en")
    c.lookup_plate.return_value = None   # safe default: plate not found
    for attr, val in overrides.items():
        getattr(c, attr).return_value = val
    return c


# ── Local fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def good_vision():
    """High-confidence detection for a plate that IS in the mock DB."""
    return {
        "plate":  "RK612AL",
        "conf":   0.92,
        "bbox":   (10, 20, 100, 60),
        "visual": np.zeros((480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def bad_vision():
    """High-confidence detection for a plate NOT in the mock DB."""
    return {
        "plate":  "XX999ZZ",
        "conf":   0.91,
        "bbox":   (10, 20, 100, 60),
        "visual": np.zeros((480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def low_conf_vision(good_vision):
    """Detection below the 0.5 confidence threshold."""
    out = dict(good_vision)
    out["conf"] = 0.35
    return out


# ── Flow A: vision confident, plate in DB ─────────────────────────────────────

class TestFlowA:

    def test_A1_success_on_name_match(self, blank_image, good_vision):
        c = _client(
            detect_plate=good_vision,
            lookup_plate=_DB_ENTRY,
            extract_name="Jane Doe",
            verify_name=True,
        )
        result, vision_out = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert result["db_entry"]["plate"] == "RK612AL"
        assert result["db_entry"]["dock"] == "Dock B"
        assert vision_out is good_vision

    def test_A1_tts_called_on_success(self, blank_image, good_vision):
        c = _client(
            detect_plate=good_vision,
            lookup_plate=_DB_ENTRY,
            extract_name="Jane Doe",
            verify_name=True,
        )
        RemoteOrchestrator(c).run_automated_entry(blank_image)
        c.synthesize.assert_called()

    def test_A2_name_mismatch_status(self, blank_image, good_vision):
        c = _client(
            detect_plate=good_vision,
            lookup_plate=_DB_ENTRY,
            extract_name="Wrong Person",
            verify_name=False,
        )
        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert result["status"] == "name_mismatch"
        assert result["plate"] == "RK612AL"

    def test_A1_vision_output_returned(self, blank_image, good_vision):
        c = _client(
            detect_plate=good_vision,
            lookup_plate=_DB_ENTRY,
            extract_name="Jane Doe",
            verify_name=True,
        )
        _, vision_out = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert vision_out is good_vision


# ── Flow B: vision confident, plate NOT in DB ─────────────────────────────────

class TestFlowB:

    def test_B1_confirmed_absent_plate_alerts_worker(self, blank_image, bad_vision):
        c = _client(detect_plate=bad_vision, parse_yes_no=True)
        # lookup always returns None — plate not in DB

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert result["status"] == "alert_worker"
        assert result["plate"] == "XX999ZZ"

    def test_B1_after_confirmation_parse_plate_not_called(self, blank_image, bad_vision):
        c = _client(detect_plate=bad_vision, parse_yes_no=True)

        RemoteOrchestrator(c).run_automated_entry(blank_image)

        # driver confirmed the reading → no audio fallback for the plate
        c.parse_plate.assert_not_called()

    def test_B2_denied_reading_audio_plate_found_name_match(self, blank_image, bad_vision):
        c = _client(detect_plate=bad_vision)
        # First lookup (bad plate) misses; second lookup (audio plate) hits
        c.lookup_plate.side_effect = [None, _DB_ENTRY]
        c.parse_yes_no.return_value = False      # "no that's wrong"
        c.parse_plate.return_value = "RK612AL"
        c.extract_name.return_value = "Jane Doe"
        c.verify_name.return_value = True

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert result["db_entry"]["driver_name"] == "Jane Doe"

    def test_B3_denied_reading_audio_plate_found_name_mismatch(self, blank_image, bad_vision):
        c = _client(detect_plate=bad_vision)
        c.lookup_plate.side_effect = [None, _DB_ENTRY]
        c.parse_yes_no.return_value = False
        c.parse_plate.return_value = "RK612AL"
        c.extract_name.return_value = "Imposter"
        c.verify_name.return_value = False

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert result["status"] == "name_mismatch"

    def test_B4_denied_reading_audio_plate_also_absent(self, blank_image, bad_vision):
        c = _client(detect_plate=bad_vision, parse_yes_no=False, parse_plate="YY000XX")
        # lookup always returns None

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"


# ── Flow C: low confidence or no detection ────────────────────────────────────

class TestFlowC:

    def test_C1_low_conf_audio_found_name_match(self, blank_image, low_conf_vision):
        c = _client(
            detect_plate=low_conf_vision,
            lookup_plate=_DB_ENTRY,
            parse_plate="RK612AL",
            extract_name="Jane Doe",
            verify_name=True,
        )
        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert result["status"] == "success"

    def test_C1_confirm_not_called_on_low_confidence(self, blank_image, low_conf_vision):
        c = _client(
            detect_plate=low_conf_vision,
            lookup_plate=_DB_ENTRY,
            parse_plate="RK612AL",
            extract_name="Jane Doe",
            verify_name=True,
        )
        RemoteOrchestrator(c).run_automated_entry(blank_image)
        # confirm_plate_with_driver uses parse_yes_no — must not be called in Flow C
        c.parse_yes_no.assert_not_called()

    def test_C2_low_conf_audio_plate_absent(self, blank_image, low_conf_vision):
        c = _client(detect_plate=low_conf_vision, parse_plate="YY000XX")

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"

    def test_C3_no_detection_audio_found_success(self, blank_image):
        c = _client(
            detect_plate=None,
            lookup_plate=_DB_ENTRY,
            parse_plate="RK612AL",
            extract_name="Jane Doe",
            verify_name=True,
        )
        result, vision_out = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert result["status"] == "success"
        assert vision_out is None   # no visual when detection returned nothing

    def test_C3_no_detection_audio_absent(self, blank_image):
        c = _client(detect_plate=None, parse_plate="YY000XX")

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert result["status"] == "alert_worker"


# ── Return type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    """RemoteGUI unpacks (result, vision_output) — this contract must always hold."""

    @pytest.mark.parametrize("status", ["success", "alert_worker", "name_mismatch"])
    def test_always_returns_two_element_tuple(self, blank_image, status):
        c = MagicMock()
        c.synthesize.return_value = (np.zeros(100, dtype=np.float32), 16_000)
        c.transcribe.return_value = ("", "en")
        c.detect_plate.return_value = None

        if status == "success":
            c.lookup_plate.return_value = _DB_ENTRY
            c.parse_plate.return_value = "RK612AL"
            c.extract_name.return_value = "Jane Doe"
            c.verify_name.return_value = True
        elif status == "name_mismatch":
            c.lookup_plate.return_value = _DB_ENTRY
            c.parse_plate.return_value = "RK612AL"
            c.extract_name.return_value = "Wrong"
            c.verify_name.return_value = False
        else:  # alert_worker
            c.lookup_plate.return_value = None
            c.parse_plate.return_value = "YY000XX"

        returned = RemoteOrchestrator(c).run_automated_entry(blank_image)

        assert isinstance(returned, tuple)
        assert len(returned) == 2
        result, _ = returned
        assert "status" in result

    def test_result_has_plate_on_alert(self, blank_image):
        c = _client(detect_plate=None, parse_plate="YY000XX")

        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert "plate" in result

    def test_result_has_db_entry_on_success(self, blank_image):
        c = _client(
            detect_plate=None,
            lookup_plate=_DB_ENTRY,
            parse_plate="RK612AL",
            extract_name="Jane Doe",
            verify_name=True,
        )
        result, _ = RemoteOrchestrator(c).run_automated_entry(blank_image)
        assert "db_entry" in result
        assert result["db_entry"]["driver_name"] == "Jane Doe"
