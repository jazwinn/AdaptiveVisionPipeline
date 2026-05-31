from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..tracking.tracker import TrackedDetection


@dataclass
class EpisodeResult:
    pipeline_name: str
    mean_confidence: float
    track_consistency: float
    flicker_rate: float
    latency_ms: float


def compute_reward(
    result: EpisodeResult,
    target_latency_ms: float = 50.0,
) -> float:
    quality = result.mean_confidence * 2.0
    stability = result.track_consistency - (result.flicker_rate * 3.0)
    cost_penalty = max(0.0, (result.latency_ms - target_latency_ms) / 100.0)
    reward = quality + stability - cost_penalty
    return float(np.clip(reward, -2.0, 3.0))


def compute_reward_image(
    result: EpisodeResult,
    target_latency_ms: float = 50.0,
) -> float:
    """Reward for independent images — omits track_consistency / flicker_rate."""
    quality = result.mean_confidence * 2.0
    cost_penalty = max(0.0, (result.latency_ms - target_latency_ms) / 100.0)
    return float(np.clip(quality - cost_penalty, -1.0, 2.0))


class WindowMetrics:
    """Accumulates per-frame data within a window and computes EpisodeResult."""

    def __init__(self):
        self._confidences: list[float] = []
        self._latencies: list[float] = []
        self._track_ids_per_frame: list[set[int]] = []
        self._id_switches = 0
        self._prev_ids: set[int] = set()

    def update(self, tracked: list[TrackedDetection], latency_ms: float):
        confs = [t.confidence for t in tracked]
        if confs:
            self._confidences.extend(confs)
        self._latencies.append(latency_ms)

        curr_ids = {t.track_id for t in tracked}
        self._track_ids_per_frame.append(curr_ids)

        # ID switches: IDs present last frame that disappeared (simple proxy)
        if self._prev_ids:
            self._id_switches += len(self._prev_ids - curr_ids)
        self._prev_ids = curr_ids

    def compute(self, pipeline_name: str) -> EpisodeResult:
        mean_conf = float(np.mean(self._confidences)) if self._confidences else 0.0
        mean_lat = float(np.mean(self._latencies)) if self._latencies else 0.0
        n_frames = len(self._track_ids_per_frame)

        # Track consistency: fraction of frames where at least one track persists
        if n_frames > 1:
            survived = sum(
                1 for i in range(1, n_frames)
                if self._track_ids_per_frame[i] & self._track_ids_per_frame[i - 1]
            )
            consistency = survived / (n_frames - 1)
        else:
            consistency = 1.0

        flicker_rate = self._id_switches / max(n_frames, 1)

        return EpisodeResult(
            pipeline_name=pipeline_name,
            mean_confidence=mean_conf,
            track_consistency=consistency,
            flicker_rate=flicker_rate,
            latency_ms=mean_lat,
        )

    def reset(self):
        self._confidences.clear()
        self._latencies.clear()
        self._track_ids_per_frame.clear()
        self._id_switches = 0
        self._prev_ids = set()
