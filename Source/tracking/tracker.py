from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np
import supervision as sv
from ..core.pipeline import Detection


@dataclass
class TrackedDetection:
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    track_id: int
    mask: np.ndarray | None = field(default=None, repr=False)
    """Boolean H×W mask inherited from the matching Detection; None for detect models."""


def _make_tracker():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return sv.ByteTrack()


class TrackerWrapper:
    def __init__(self):
        self.tracker = _make_tracker()
        self._class_names: dict[int, str] = {}

    def update(self, dets: list[Detection]) -> list[TrackedDetection]:
        for d in dets:
            self._class_names[d.class_id] = d.class_name

        if not dets:
            empty = sv.Detections.empty()
            tracked = self.tracker.update_with_detections(empty)
        else:
            sv_dets = sv.Detections(
                xyxy=np.array([d.bbox_xyxy for d in dets]),
                confidence=np.array([d.confidence for d in dets]),
                class_id=np.array([d.class_id for d in dets]),
            )
            tracked = self.tracker.update_with_detections(sv_dets)

        has_masks = any(d.mask is not None for d in dets)
        results = []
        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            # Thread the mask from the closest input detection (by bbox center distance).
            # ByteTrack may reorder detections or add Kalman-predicted positions, so we
            # match by nearest center rather than by index.
            mask = self._nearest_mask(tracked.xyxy[i], dets) if has_masks else None
            results.append(TrackedDetection(
                bbox_xyxy=tracked.xyxy[i],
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                class_id=class_id,
                class_name=self._class_names.get(class_id, "unknown"),
                track_id=int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1,
                mask=mask,
            ))
        return results

    @staticmethod
    def _nearest_mask(
        bbox: np.ndarray,
        dets: list[Detection],
    ) -> np.ndarray | None:
        """Return the mask of the Detection whose center is closest to bbox."""
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        best_dist = float("inf")
        best_mask: np.ndarray | None = None
        for d in dets:
            dcx = (d.bbox_xyxy[0] + d.bbox_xyxy[2]) * 0.5
            dcy = (d.bbox_xyxy[1] + d.bbox_xyxy[3]) * 0.5
            dist = (cx - dcx) ** 2 + (cy - dcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_mask = d.mask
        return best_mask

    def reset(self):
        self.tracker = _make_tracker()
