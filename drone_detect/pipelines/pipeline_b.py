from __future__ import annotations
import numpy as np
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection


class PipelineB(DetectionPipeline):
    """Accuracy baseline: medium YOLOv8m at 1280px, no preprocessing."""

    name = "high_res"
    cost_estimate = 4.0

    def __init__(self, model_path: str = "yolov8m.pt", imgsz: int = 1280, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return frame

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
