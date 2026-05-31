from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def apply_denoise(image: np.ndarray) -> np.ndarray:
    """Non-Local Means denoising: smooths sensor/Gaussian noise while preserving edges."""
    return cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)


class PipelineF(DetectionPipeline):
    """Noise pipeline: Non-Local Means denoising then YOLOv8n."""

    name = "denoise_pipeline"
    cost_estimate = 2.5

    def __init__(self, model_path: str | None = None, imgsz: int = 640, conf: float = 0.3):
        if model_path is None:
            model_path = str(_MODELS_DIR / "yolov8n.pt")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return apply_denoise(frame)

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
