import logging
import numpy as np
import cv2
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
from typing import Dict, Any, Optional

import yaml
logger = logging.getLogger(__name__)

def read_config(path: str) -> Dict[str, Any]:
    """Read the YAML config file at `path` and return it as a dict."""
    
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

class VisionPipeline:
    def __init__(self, config: Dict[str, Any], device: str = "cpu", onnx: bool = False) -> None:
        if onnx:
            self.vision_model: YOLO = YOLO(model=config["models"]["model_path_onnx"], task="detect")
        else:
            self.vision_model: YOLO = YOLO(model=config["models"]["model_path"], task="detect")
        self.recognizer: LicensePlateRecognizer = LicensePlateRecognizer(
            hub_ocr_model="cct-s-v2-global-model", device=device
        )

    def _select_best_box(self, boxes):
        """Return the box with the highest YOLO detection confidence.
        When multiple plates are detected, logs a warning so it's visible in logs.
        """
        if len(boxes) == 1:
            return boxes[0]
        best = max(boxes, key=lambda b: float(b.conf[0]))
        logger.warning(
            f"{len(boxes)} plates detected — using the highest-confidence box "
            f"(det_conf={float(best.conf[0]):.2f})"
        )
        return best

    def read_plate(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect and OCR the license plate in `image`.

        Args:
            image: BGR image array as returned by cv2.imread().

        Returns:
            A dict with:
                plate   (str)          recognised plate text
                conf    (float)        mean OCR character confidence [0, 1]
                bbox    (tuple)        (x1, y1, x2, y2) pixel coords of the plate region
                visual  (np.ndarray)   annotated image with YOLO bounding box drawn
            or None if no plate was detected in the image.
        """
        results = self.vision_model(image)

        if not results or len(results[0].boxes) == 0:
            logger.info("No license plate detected.")
            return None

        box = self._select_best_box(results[0].boxes)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning("Bounding box crop is empty — skipping OCR.")
            return None

        try:
            prediction = self.recognizer.run(crop, return_confidence=True)[0]
        except Exception:
            logger.exception("OCR failed on cropped plate region.")
            return None

        conf = float(np.mean(prediction.char_probs))
        logger.info(f"Plate: {prediction.plate!r}  OCR conf: {conf:.2f}")

        return {
            "plate": prediction.plate,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "visual": results[0].plot(),   # annotated BGR array
        }
