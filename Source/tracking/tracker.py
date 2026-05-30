from __future__ import annotations
import warnings
from dataclasses import dataclass
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

        results = []
        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            results.append(TrackedDetection(
                bbox_xyxy=tracked.xyxy[i],
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                class_id=class_id,
                class_name=self._class_names.get(class_id, "unknown"),
                track_id=int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1,
            ))
        return results

    def reset(self):
        self.tracker = _make_tracker()
