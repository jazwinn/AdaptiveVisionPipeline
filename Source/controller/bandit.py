from __future__ import annotations
import numpy as np
from .base import MetaController
from ..features.extractor import FeatureVector

FEATURE_DIM = 12


def _to_vec(f: FeatureVector) -> np.ndarray:
    return np.array([
        f.laplacian_variance / 1000.0,
        f.mean_intensity / 255.0,
        f.optical_flow_magnitude / 20.0,
        f.mean_confidence,
        f.small_object_ratio,
        f.edge_density,
        f.entropy / 8.0,
        f.detection_count / 50.0,
        f.underexposed_ratio,
        f.overexposed_ratio,
        f.fft_blur_score / 100.0,
        f.frame_displacement / 30.0,
    ], dtype=np.float64)


class UCBBanditController(MetaController):
    """Upper-confidence-bound multi-armed bandit."""

    window_size = 30

    def __init__(self, pipeline_names: list[str], c: float = 1.4):
        self.names = pipeline_names
        self.c = c
        self.counts: dict[str, int] = {n: 0 for n in pipeline_names}
        self.values: dict[str, float] = {n: 0.5 for n in pipeline_names}
        self.total_pulls = 0

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        self.total_pulls += 1
        for name in pipeline_names:
            if self.counts[name] == 0:
                return name

        ucb_scores = {}
        for name in pipeline_names:
            exploitation = self.values[name]
            exploration = self.c * np.sqrt(np.log(self.total_pulls) / self.counts[name])
            ucb_scores[name] = exploitation + exploration
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, pipeline_name: str, reward: float, features: FeatureVector | None = None):
        self.counts[pipeline_name] += 1
        n = self.counts[pipeline_name]
        self.values[pipeline_name] += (reward - self.values[pipeline_name]) / n

    def summary(self) -> dict:
        return {
            "counts": dict(self.counts),
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "total_pulls": self.total_pulls,
        }


class ContextualBanditController(MetaController):
    """Linear contextual bandit with UCB-style exploration."""

    window_size = 30

    def __init__(self, pipeline_names: list[str], lr: float = 0.01, c: float = 0.5):
        self.names = pipeline_names
        self.lr = lr
        self.c = c
        self.weights: dict[str, np.ndarray] = {
            n: np.zeros(FEATURE_DIM) for n in pipeline_names
        }
        self.counts: dict[str, int] = {n: 0 for n in pipeline_names}
        self.total_pulls = 0

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        self.total_pulls += 1
        fvec = _to_vec(feature_history[-1])

        # Ensure unexplored pipelines are tried first
        for name in pipeline_names:
            if self.counts[name] == 0:
                return name

        scores = {}
        for name in pipeline_names:
            exploitation = float(self.weights[name] @ fvec)
            exploration = self.c * np.sqrt(np.log(self.total_pulls) / self.counts[name])
            scores[name] = exploitation + exploration
        return max(scores, key=scores.get)

    def update(self, pipeline_name: str, reward: float, features: FeatureVector | None = None):
        self.counts[pipeline_name] += 1
        if features is None:
            return
        fvec = _to_vec(features)
        pred = float(self.weights[pipeline_name] @ fvec)
        error = reward - pred
        self.weights[pipeline_name] += self.lr * error * fvec
