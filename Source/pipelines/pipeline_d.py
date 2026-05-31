from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class PipelineD(DetectionPipeline):
    """Low-light pipeline: fast YOLOv8n with CLAHE preprocessing."""

    name = "clahe_pipeline"
    cost_estimate = 1.5

    def __init__(self, model_path: str | None = None, imgsz: int = 640, conf: float = 0.3):
        if model_path is None:
            model_path = str(_MODELS_DIR / "yolov8n.pt")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return apply_clahe(frame)

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
