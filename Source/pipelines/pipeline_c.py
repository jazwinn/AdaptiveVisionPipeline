from __future__ import annotations
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection
from ..core.config import TilingConfig


class PipelineC(DetectionPipeline):
    """Tiled inference: 640px tiles with overlap, best for small objects."""

    name = "tiled"
    cost_estimate = 4.0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        tiling_config: TilingConfig | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.cfg = tiling_config or TilingConfig()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def generate_tiles(self, image: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        h, w = image.shape[:2]
        step = int(self.cfg.tile_size * (1 - self.cfg.overlap))
        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + self.cfg.tile_size, w)
                y2 = min(y + self.cfg.tile_size, h)
                tile = image[y:y2, x:x2]
                tiles.append((tile, x, y))
                if x2 == w:
                    break
                x += step
            if y + self.cfg.tile_size >= h:
                break
            y += step
        return tiles

    def _infer_tile(self, tile: np.ndarray) -> list[Detection]:
        results = self.model(tile, imgsz=self.cfg.tile_size, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append(Detection(
                bbox_xyxy=box.xyxy.cpu().numpy()[0],
                confidence=float(box.conf),
                class_id=int(box.cls),
                class_name=results.names[int(box.cls)],
            ))
        return detections

    def _nms_merge(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        boxes = np.array([d.bbox_xyxy for d in dets])
        scores = np.array([d.confidence for d in dets])
        class_ids = np.array([d.class_id for d in dets])
        sv_dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        sv_dets = sv_dets.with_nms(threshold=0.5)
        return [
            Detection(
                bbox_xyxy=sv_dets.xyxy[i],
                confidence=float(sv_dets.confidence[i]),
                class_id=int(sv_dets.class_id[i]),
                class_name=dets[0].class_name,
            )
            for i in range(len(sv_dets))
        ]

    def infer(self, image: np.ndarray) -> list[Detection]:
        all_dets: list[Detection] = []
        for tile, ox, oy in self.generate_tiles(image):
            for d in self._infer_tile(tile):
                shifted = Detection(
                    bbox_xyxy=d.bbox_xyxy + np.array([ox, oy, ox, oy]),
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                )
                all_dets.append(shifted)
        return self._nms_merge(all_dets)
