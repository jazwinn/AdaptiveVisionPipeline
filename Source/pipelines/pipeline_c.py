from __future__ import annotations
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ..core.pipeline import DetectionPipeline, Detection, yolo_results_to_detections
from ..core.config import TilingConfig

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class PipelineC(DetectionPipeline):
    """Tiled inference: 640px tiles with overlap, best for small objects."""

    name = "tiled"
    cost_estimate = 4.0

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = 0.25,
        tiling_config: TilingConfig | None = None,
        imgsz: int = 640,   # accepted for API consistency; tile size is set via TilingConfig
    ):
        if model_path is None:
            model_path = str(_MODELS_DIR / "yolov8n.pt")
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

    def _infer_tile(
        self,
        tile: np.ndarray,
        ox: int,
        oy: int,
        full_h: int,
        full_w: int,
    ) -> list[Detection]:
        results = self.model(tile, imgsz=self.cfg.tile_size, conf=self.conf, verbose=False)[0]
        tile_dets = yolo_results_to_detections(results)
        shifted: list[Detection] = []
        for d in tile_dets:
            # Embed tile-level mask into full-image canvas
            full_mask: np.ndarray | None = None
            if d.mask is not None:
                full_mask = np.zeros((full_h, full_w), dtype=bool)
                th, tw = d.mask.shape
                y2 = min(oy + th, full_h)
                x2 = min(ox + tw, full_w)
                full_mask[oy:y2, ox:x2] = d.mask[:y2 - oy, :x2 - ox]
            shifted.append(Detection(
                bbox_xyxy=d.bbox_xyxy + np.array([ox, oy, ox, oy]),
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                mask=full_mask,
            ))
        return shifted

    def _nms_merge(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        boxes     = np.array([d.bbox_xyxy for d in dets])
        scores    = np.array([d.confidence for d in dets])
        class_ids = np.array([d.class_id for d in dets])
        sv_dets   = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        sv_dets   = sv_dets.with_nms(threshold=0.5)
        merged: list[Detection] = []
        for i in range(len(sv_dets)):
            # Find the original detection with the closest bbox to recover its mask
            kept_box = sv_dets.xyxy[i]
            best = min(dets, key=lambda d: float(np.sum((d.bbox_xyxy - kept_box) ** 2)))
            merged.append(Detection(
                bbox_xyxy=sv_dets.xyxy[i],
                confidence=float(sv_dets.confidence[i]),
                class_id=int(sv_dets.class_id[i]),
                class_name=best.class_name,
                mask=best.mask,
            ))
        return merged

    def infer(self, image: np.ndarray) -> list[Detection]:
        fh, fw = image.shape[:2]
        all_dets: list[Detection] = []
        for tile, ox, oy in self.generate_tiles(image):
            all_dets.extend(self._infer_tile(tile, ox, oy, fh, fw))
        return self._nms_merge(all_dets)
