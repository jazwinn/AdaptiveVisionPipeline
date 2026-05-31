from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def apply_gamma(image: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_bright_correction(image: np.ndarray) -> np.ndarray:
    """Gamma compression to recover blown-out highlights, then mild CLAHE for contrast."""
    img = apply_gamma(image, gamma=0.55)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class PipelineE(DetectionPipeline):
    """Overexposure / glare pipeline: gamma correction + mild CLAHE, then YOLOv8n."""

    name = "bright_pipeline"
    cost_estimate = 1.5

    def __init__(self, model_path: str | None = None, imgsz: int = 640, conf: float = 0.3):
        if model_path is None:
            model_path = str(_MODELS_DIR / "yolov8n.pt")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return apply_bright_correction(frame)

    def infer(self, image: np.ndarray) -> list[Detection]:
        results = self.model(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append(Detection(
                bbox_xyxy=box.xyxy.cpu().numpy()[0],
                confidence=float(box.conf),
                class_id=int(box.cls),
                class_name=results.names[int(box.cls)],
            ))
        return detections
