"""
Neural Network (MLP Classifier) meta-controller.

A supervised pipeline-selection controller backed by a PyTorch MLP trained
offline from replay-buffer data.  Identical training surface to Decision Tree /
Random Forest but uses deep-learning non-linearities and mini-batch SGD.

Falls back gracefully to rule-based logic if:
  - PyTorch is not installed
  - No trained model file exists at the expected path
"""
from __future__ import annotations

import dataclasses
import sys
import warnings as _warnings
from pathlib import Path
from typing import Any

import numpy as np

from .base import MetaController
from ..features.extractor import FeatureVector

# ── Feature names (stable order from dataclass fields) ───────────────────────

FEATURE_NAMES: list[str] = [f.name for f in dataclasses.fields(FeatureVector)]
AGG_FEATURE_NAMES: list[str] = (
    [f"mean_{f}" for f in FEATURE_NAMES] + [f"std_{f}" for f in FEATURE_NAMES]
)

_PENDING_MAX = 10_000

# ── Optional torch import ─────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Network architecture ──────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _MLPClassifier(nn.Module):
        """Small MLP classifier: 26 → 128 → 64 → n_classes."""

        def __init__(self, input_dim: int, n_classes: int, hidden_dims: list[int]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            dropouts = [0.3, 0.2] + [0.1] * max(0, len(hidden_dims) - 2)
            for h, d in zip(hidden_dims, dropouts):
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)
else:
    _MLPClassifier = None  # type: ignore[assignment,misc]


# ── Controller ────────────────────────────────────────────────────────────────

class NeuralNetController(MetaController):
    """
    Supervised MLP classifier for pipeline selection.

    Trained offline by ``train_nn.py``.  During live inference the controller
    runs a forward pass on the 26-dim aggregated feature vector and returns the
    argmax class.  If the model file is missing or PyTorch is unavailable, the
    controller falls back to the same rule-based heuristics used by
    ``RuleBasedController``.
    """

    MODEL_FILENAME = "nn_controller.pt"
    window_size: int = 30

    def __init__(
        self,
        pipeline_names: list[str],
        model_path: Path | None = None,
    ) -> None:
        self.pipeline_names = list(pipeline_names)
        self._model: Any = None
        self._pipeline_classes: list[str] = []
        self._pending: list[tuple[np.ndarray, str, float]] = []

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / self.MODEL_FILENAME
        self._model_path = Path(model_path)

        if not _TORCH_AVAILABLE:
            print(
                "[NeuralNetController] PyTorch not installed -using rule-based fallback.",
                file=sys.stderr,
            )
        else:
            self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if not self._model_path.exists():
            print(
                f"[NeuralNetController] Model not found at {self._model_path} "
                f"— using rule-based fallback.  Run: python -m Source.controller.train_nn",
                file=sys.stderr,
            )
            return
        try:
            artifact = torch.load(  # type: ignore[union-attr]
                self._model_path, map_location="cpu", weights_only=False
            )
            hidden_dims: list[int] = artifact.get("hidden_dims", [128, 64])
            pipeline_classes: list[str] = artifact["pipeline_classes"]
            n_classes = len(pipeline_classes)

            net = _MLPClassifier(  # type: ignore[call-arg]
                input_dim=artifact.get("input_dim", 26),
                n_classes=n_classes,
                hidden_dims=hidden_dims,
            )
            raw_sd = artifact["model_state_dict"]
            # Handle state dicts saved from a bare nn.Sequential (keys like "0.weight")
            # vs ones saved from a module with self.net (keys like "net.0.weight").
            if any(k.startswith("net.") for k in raw_sd):
                state_dict = raw_sd
            else:
                state_dict = {"net." + k: v for k, v in raw_sd.items()}
            net.load_state_dict(state_dict)
            net.eval()

            self._model = net
            self._pipeline_classes = pipeline_classes
        except Exception as exc:
            _warnings.warn(
                f"[NeuralNetController] Failed to load model from {self._model_path}: {exc}. "
                "Using rule-based fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

    # ── MetaController interface ───────────────────────────────────────────

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        if not feature_history:
            return pipeline_names[0]

        if self._model is None or not _TORCH_AVAILABLE:
            return self._rule_fallback(feature_history[-1], pipeline_names)

        vec = aggregate_history(feature_history)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # type: ignore[union-attr]

        with torch.no_grad():  # type: ignore[union-attr]
            logits = self._model(x)
            pred_idx = int(logits.argmax(dim=1).item())

        if pred_idx < len(self._pipeline_classes):
            prediction = self._pipeline_classes[pred_idx]
            if prediction in pipeline_names:
                return prediction

        # Fallback: predicted class not in available pipelines
        return pipeline_names[0]

    def update(
        self,
        pipeline_name: str,
        reward: float,
        features: FeatureVector | None = None,
    ) -> None:
        """Accumulate experience for potential offline re-training."""
        if features is not None and len(self._pending) < _PENDING_MAX:
            self._pending.append(
                (aggregate_history([features]), pipeline_name, reward)
            )

    # ── Rule-based fallback ────────────────────────────────────────────────

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
