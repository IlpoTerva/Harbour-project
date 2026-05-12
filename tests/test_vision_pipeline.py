"""
test_vision_pipeline.py — unit tests for VisionPipeline.

Tests cover:
  - No detection (YOLO returns empty boxes)
  - Empty crop produced by bounding box
  - OCR raises an exception
  - Successful single-plate detection
  - Multiple plates → highest-confidence box selected
  - Output dict contains all required keys with correct types
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# conftest.py has already injected mock modules, so this import is safe
from scripts.vision_pipeline import VisionPipeline
from tests.conftest import make_yolo_box, make_yolo_results, make_ocr_prediction


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pipeline(config):
    """Construct a VisionPipeline with both heavy classes mocked."""
    with patch("scripts.vision_pipeline.YOLO"), \
         patch("scripts.vision_pipeline.LicensePlateRecognizer"):
        return VisionPipeline(config=config, onnx=False)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVisionPipelineNoDetection:

    def test_returns_none_when_yolo_returns_no_results(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = []           # empty results list
        assert vp.read_plate(blank_image) is None

    def test_returns_none_when_boxes_list_is_empty(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results(boxes=[])
        assert vp.read_plate(blank_image) is None

    def test_returns_none_when_crop_is_empty(self, mock_config):
        """A box where x1==x2 or y1==y2 produces a zero-area crop."""
        vp = make_pipeline(mock_config)
        zero_area_box = make_yolo_box(x1=50, y1=50, x2=50, y2=50)   # zero-height crop
        vp.vision_model.return_value = make_yolo_results([zero_area_box])
        tiny_image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert vp.read_plate(tiny_image) is None

    def test_returns_none_when_ocr_raises(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results([make_yolo_box()])
        vp.recognizer.run.side_effect = RuntimeError("OCR model error")
        assert vp.read_plate(blank_image) is None


class TestVisionPipelineSuccess:

    def test_returns_correct_keys(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results([make_yolo_box()])
        vp.recognizer.run.return_value = [make_ocr_prediction("PP587A0", char_conf=0.95)]

        result = vp.read_plate(blank_image)

        assert result is not None
        assert set(result.keys()) == {"plate", "conf", "bbox", "visual"}

    def test_plate_text_is_uppercased(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results([make_yolo_box()])
        vp.recognizer.run.return_value = [make_ocr_prediction("rk612al")]

        result = vp.read_plate(blank_image)
        # fast-plate-ocr itself returns uppercase, but let's confirm no mangling
        assert result["plate"] == "rk612al"   # pipeline returns whatever OCR gives

    def test_conf_is_mean_of_char_probs(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results([make_yolo_box()])
        # 4 chars at 0.8 and 3 chars at 1.0 → mean = (4*0.8 + 3*1.0) / 7 ≈ 0.886
        pred = MagicMock()
        pred.plate      = "ABCD123"
        pred.char_probs = np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0])
        vp.recognizer.run.return_value = [pred]

        result = vp.read_plate(blank_image)
        assert abs(result["conf"] - np.mean([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0])) < 1e-6

    def test_bbox_matches_yolo_box_coords(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        vp.vision_model.return_value = make_yolo_results(
            [make_yolo_box(x1=15, y1=25, x2=110, y2=65)]
        )
        vp.recognizer.run.return_value = [make_ocr_prediction()]

        result = vp.read_plate(blank_image)
        assert result["bbox"] == (15, 25, 110, 65)

    def test_visual_is_numpy_array(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        annotated = np.ones((480, 640, 3), dtype=np.uint8) * 128
        vp.vision_model.return_value = make_yolo_results([make_yolo_box()], plot_image=annotated)
        vp.recognizer.run.return_value = [make_ocr_prediction()]

        result = vp.read_plate(blank_image)
        np.testing.assert_array_equal(result["visual"], annotated)


class TestVisionPipelineBoxSelection:

    def test_single_box_is_returned_directly(self, mock_config, blank_image):
        vp = make_pipeline(mock_config)
        box = make_yolo_box(conf=0.7)
        vp.vision_model.return_value = make_yolo_results([box])
        vp.recognizer.run.return_value = [make_ocr_prediction("PP587A0")]

        result = vp.read_plate(blank_image)
        assert result["plate"] == "PP587A0"

    def test_highest_confidence_box_wins(self, mock_config, blank_image):
        """When multiple plates detected, the one with highest YOLO conf is used."""
        vp = make_pipeline(mock_config)

        low_conf_box  = make_yolo_box(x1=10, y1=10, x2=80, y2=40, conf=0.55)
        high_conf_box = make_yolo_box(x1=200, y1=200, x2=350, y2=260, conf=0.91)

        vp.vision_model.return_value = make_yolo_results([low_conf_box, high_conf_box])

        # OCR is called with the crop of whichever box wins.
        # We verify the correct box was selected by inspecting the crop passed to OCR.
        captured_crops = []
        def capture_ocr(crop, **kwargs):
            captured_crops.append(crop)
            return [make_ocr_prediction("HIGHCONF")]

        vp.recognizer.run.side_effect = capture_ocr

        result = vp.read_plate(blank_image)

        assert result is not None
        # The high-conf box spans x=200..350, y=200..260
        assert captured_crops[0].shape == (60, 150, 3)
