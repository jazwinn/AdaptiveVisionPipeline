from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
from .frame_reader import Frame


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = field(default=None, repr=False)
    """Boolean H×W mask (original image resolution).
    Populated by segmentation models; None for detection-only models."""


def yolo_results_to_detections(results) -> list[Detection]:
    """
    Convert an ultralytics Results object to a list of Detection.

    Handles both detection models (results.masks is None) and segmentation
    models (results.masks.data has shape N×H×W float32, converted to bool).
    """
    detections: list[Detection] = []
    masks_obj = results.masks  # None for detect models
    # Original image dimensions (the model runs on a letterboxed/downscaled copy,
    # so masks.data comes back at the network's mask resolution, NOT this size).
    orig_h, orig_w = results.orig_shape  # (h, w)
    for i, box in enumerate(results.boxes):
        mask: np.ndarray | None = None
        if masks_obj is not None and i < len(masks_obj.data):
            # masks.data: (N, mh, mw) float32 at network mask resolution → CPU.
            m = masks_obj.data[i].cpu().numpy()
            # Resize to the ORIGINAL image resolution so it overlays correctly.
            if m.shape[0] != orig_h or m.shape[1] != orig_w:
                m = cv2.resize(
                    m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
            mask = m.astype(bool)
        detections.append(Detection(
            bbox_xyxy=box.xyxy.cpu().numpy()[0],
            confidence=float(box.conf),
            class_id=int(box.cls),
            class_name=results.names[int(box.cls)],
            mask=mask,
        ))
    return detections


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
