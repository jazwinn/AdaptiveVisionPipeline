from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from .base import MetaController
from ..features.extractor import FeatureVector

FEATURE_NAMES: list[str] = [f.name for f in dataclasses.fields(FeatureVector)]
AGG_FEATURE_NAMES: list[str] = (
    [f"mean_{f}" for f in FEATURE_NAMES] + [f"std_{f}" for f in FEATURE_NAMES]
)

_PENDING_MAX = 10_000


def aggregate_history(feature_history: list[FeatureVector]) -> np.ndarray:
    """Reduce list[FeatureVector] → 26-dim vector (mean + std per feature)."""
    if not feature_history:
        return np.zeros(26, dtype=np.float64)
    rows = np.array(
        [[getattr(fv, name) for name in FEATURE_NAMES] for fv in feature_history],
        dtype=np.float64,
    )
    means = rows.mean(axis=0)
    stds = rows.std(axis=0) if len(rows) > 1 else np.zeros(len(FEATURE_NAMES))
    return np.concatenate([means, stds])


class DecisionTreeController(MetaController):
    MODEL_FILENAME = "dt_controller.joblib"
    window_size: int = 30

    def __init__(
        self,
        pipeline_names: list[str],
        model_path: Path | None = None,
    ) -> None:
        self.pipeline_names = pipeline_names
        self._model: Any = None
        self._selected_features: list[int] | None = None
        self._pending: list[tuple[np.ndarray, str, float]] = []

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / self.MODEL_FILENAME
        self._model_path = Path(model_path)
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            return
        try:
            import joblib
            artifact = joblib.load(self._model_path)
            self._model = artifact["model"]
            self._selected_features = artifact.get("selected_features")
        except Exception:
            self._model = None

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        if not feature_history:
            return pipeline_names[0]

        if self._model is None:
            return self._rule_fallback(feature_history[-1], pipeline_names)

        vec = aggregate_history(feature_history)
        if self._selected_features is not None:
            vec = vec[self._selected_features]

        prediction: str = self._model.predict(vec.reshape(1, -1))[0]
        return prediction if prediction in pipeline_names else pipeline_names[0]

    def update(
        self,
        pipeline_name: str,
        reward: float,
        features: FeatureVector | None = None,
    ) -> None:
        if features is not None and len(self._pending) < _PENDING_MAX:
            self._pending.append((aggregate_history([features]), pipeline_name, reward))

    def reload(self) -> None:
        """Reload model from disk (e.g. after retraining). Safe to call at any time."""
        self._model = None
        self._selected_features = None
        self._load_model()

    def export_pending(self, replay_path: str) -> int:
        """
        Write accumulated experience from ``_pending`` to the global replay buffer.

        Returns the number of entries written and clears ``_pending``.
        """
        import json
        import time as _time

        if not self._pending:
            return 0
        p = Path(replay_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            for vec, pipeline, reward in self._pending:
                feat_dict = {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}
                entry = {
                    "timestamp": _time.time(),
                    "features":  feat_dict,
                    "pipeline":  pipeline,
                    "reward":    float(reward),
                }
                fh.write(json.dumps(entry) + "\n")
        n = len(self._pending)
        self._pending.clear()
        return n

    def _rule_fallback(self, f: FeatureVector, pipeline_names: list[str]) -> str:
        def _avail(name: str) -> str:
            return name if name in pipeline_names else pipeline_names[0]

        if f.mean_intensity < 60 and f.intensity_std < 25:
            return _avail("clahe_pipeline")
        if f.optical_flow_magnitude > 8.0:
            return _avail("fast_baseline")
        if f.small_object_ratio > 0.5 or f.edge_density > 0.15:
            return _avail("tiled")
        if f.mean_confidence < 0.35 and f.detection_count < 3:
            return _avail("high_res")
        return _avail("fast_baseline")


class RandomForestController(DecisionTreeController):
    MODEL_FILENAME = "rf_controller.joblib"
