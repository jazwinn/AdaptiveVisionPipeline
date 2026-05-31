"""
Deep Q-Network (DQN) Reinforcement Learning meta-controller.

Learns a Q-function Q(state, action) over the 26-dim aggregated feature state
and discrete pipeline actions.  Supports:

  - Offline pre-training from the replay buffer via ``train_rl.py``
  - Optional online fine-tuning during live inference (epsilon-greedy)

Falls back gracefully to rule-based logic if:
  - PyTorch is not installed
  - No trained model file exists at the expected path
"""
from __future__ import annotations

import dataclasses
import sys
import warnings as _warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .base import MetaController
from ..features.extractor import FeatureVector

# ── Feature names ─────────────────────────────────────────────────────────────

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
    class _QNetwork(nn.Module):
        """MLP Q-function: state → Q-values for each discrete action."""

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: list[int],
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = state_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)
else:
    _QNetwork = None  # type: ignore[assignment,misc]


# ── Experience replay buffer ──────────────────────────────────────────────────

class _ReplayBuffer:
    """Fixed-capacity FIFO ring buffer for DQN experience replay."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
    ) -> None:
        self._buf.append((s.astype(np.float32), int(a), float(r), s_next.astype(np.float32)))

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self._buf), size=batch_size)
        states, actions, rewards, next_states = zip(*[self._buf[i] for i in idx])
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── Controller ────────────────────────────────────────────────────────────────

class NeuralRLController(MetaController):
    """
    DQN-based reinforcement learning meta-controller.

    Pre-train offline::

        python -m Source.controller.train_rl

    Then select ``neural_rl`` in the GUI dropdown.  The model loads from
    ``Source/controller/models/rl_controller.pt`` automatically.

    For online fine-tuning during live inference pass ``online_finetune=True``.
    The controller will perform mini-batch gradient steps in the background
    as experience accumulates.
    """

    MODEL_FILENAME = "rl_controller.pt"
    window_size: int = 30

    def __init__(
        self,
        pipeline_names: list[str],
        model_path: Path | None = None,
        online_finetune: bool = True,
        epsilon_start: float = 0.15,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ) -> None:
        self.pipeline_names = list(pipeline_names)
        self._online_finetune = online_finetune

        # Q-networks (set in _load_model)
        self._q_online: Any = None
        self._q_target: Any = None
        self._optimizer: Any = None
        self._pipeline_names_from_model: list[str] = []

        # Exploration
        self._epsilon = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        # Online training hyper-params
        self._gamma = gamma
        self._lr = lr
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._train_steps = 0

        # Online experience buffer
        self._replay = _ReplayBuffer(capacity=_PENDING_MAX)
        self._rng = np.random.default_rng(0)

        # For building transitions: remember last state
        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / self.MODEL_FILENAME
        self._model_path = Path(model_path)

        if not _TORCH_AVAILABLE:
            print(
                "[NeuralRLController] PyTorch not installed -using rule-based fallback.",
                file=sys.stderr,
            )
        else:
            self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if not self._model_path.exists():
            print(
                f"[NeuralRLController] Model not found at {self._model_path} "
                f"— using rule-based fallback.  Run: python -m Source.controller.train_rl",
                file=sys.stderr,
            )
            return
        try:
            artifact = torch.load(
                self._model_path, map_location="cpu", weights_only=False
            )
            hidden_dims: list[int] = artifact.get("hidden_dims", [128, 128, 64])
            pipeline_names_from_model: list[str] = artifact["pipeline_names"]
            state_dim: int = artifact.get("state_dim", 26)
            action_dim: int = artifact.get("action_dim", len(pipeline_names_from_model))

            raw_sd = artifact["q_state_dict"]
            # Handle state dicts saved from a bare nn.Sequential (keys like "0.weight")
            # vs ones saved from a module with self.net (keys like "net.0.weight").
            if any(k.startswith("net.") for k in raw_sd):
                state_dict = raw_sd
            else:
                state_dict = {"net." + k: v for k, v in raw_sd.items()}

            q_online = _QNetwork(state_dim, action_dim, hidden_dims)
            q_online.load_state_dict(state_dict)
            q_online.eval()

            q_target = _QNetwork(state_dim, action_dim, hidden_dims)
            q_target.load_state_dict(state_dict)
            q_target.eval()

            self._q_online = q_online
            self._q_target = q_target
            self._pipeline_names_from_model = pipeline_names_from_model

            # Restore saved epsilon (so resumed fine-tuning continues decaying)
            self._epsilon = artifact.get("epsilon", self._epsilon)

            if self._online_finetune:
                self._optimizer = torch.optim.Adam(
                    self._q_online.parameters(), lr=self._lr
                )

        except Exception as exc:
            _warnings.warn(
                f"[NeuralRLController] Failed to load model from {self._model_path}: {exc}. "
                "Using rule-based fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._q_online = None
            self._q_target = None

    # ── MetaController interface ───────────────────────────────────────────

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        if not feature_history:
            return pipeline_names[0]

        if self._q_online is None or not _TORCH_AVAILABLE:
            return self._rule_fallback(feature_history[-1], pipeline_names)

        state = aggregate_history(feature_history).astype(np.float32)

        # Epsilon-greedy (only during online fine-tuning)
        if self._online_finetune and self._rng.random() < self._epsilon:
            action_idx = int(self._rng.integers(0, len(self._pipeline_names_from_model)))
        else:
            x = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self._q_online(x)
            action_idx = int(q_vals.argmax(dim=1).item())

        # Store state for next update() call
        self._last_state = state
        self._last_action = action_idx

        # Map action index → pipeline name
        if action_idx < len(self._pipeline_names_from_model):
            candidate = self._pipeline_names_from_model[action_idx]
            if candidate in pipeline_names:
                return candidate

        return pipeline_names[0]

    def update(
        self,
        pipeline_name: str,
        reward: float,
        features: FeatureVector | None = None,
    ) -> None:
        """Store transition and optionally run a gradient step."""
        if not _TORCH_AVAILABLE or self._q_online is None:
            return
        if self._last_state is None or self._last_action is None:
            return

        # Build next-state
        if features is not None:
            next_state = aggregate_history([features]).astype(np.float32)
        else:
            next_state = self._last_state.copy()

        self._replay.push(self._last_state, self._last_action, reward, next_state)

        # Decay exploration
        if self._online_finetune:
            self._epsilon = max(
                self._epsilon_min, self._epsilon * self._epsilon_decay
            )
            if len(self._replay) >= self._batch_size:
                self._train_step()

        # Reset for next window
        self._last_state = None
        self._last_action = None

    # ── Online training step ───────────────────────────────────────────────

    def _train_step(self) -> None:
        if self._optimizer is None:
            return

        states, actions, rewards, next_states = self._replay.sample(
            self._batch_size, self._rng
        )

        s_t  = torch.from_numpy(states)
        a_t  = torch.from_numpy(actions)
        r_t  = torch.from_numpy(rewards)
        ns_t = torch.from_numpy(next_states)

        # Current Q-values
        q_vals = self._q_online(s_t)
        q_pred = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Bellman target using frozen target network
        with torch.no_grad():
            q_next = self._q_target(ns_t).max(dim=1).values
            q_target_val = r_t + self._gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_pred, q_target_val)
        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self._q_online.parameters(), max_norm=1.0)
        self._optimizer.step()

        self._train_steps += 1

        # Hard target-network update
        if self._train_steps % self._target_update_freq == 0:
            self._q_target.load_state_dict(self._q_online.state_dict())
            self._q_target.eval()

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
