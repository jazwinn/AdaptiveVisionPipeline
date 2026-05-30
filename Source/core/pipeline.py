from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import numpy as np
from .frame_reader import Frame


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str


class DetectionPipeline(ABC):
    name: str = "base"
    cost_estimate: float = 1.0

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def infer(self, image: np.ndarray) -> list[Detection]: ...

    def run(self, frame: Frame) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        img = self.preprocess(frame.image)
        dets = self.infer(img)
        elapsed = time.perf_counter() - t0
        meta = {
            "pipeline": self.name,
            "latency_ms": elapsed * 1000,
            "n_detections": len(dets),
        }
        return dets, meta
