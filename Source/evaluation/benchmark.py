"""
Controller benchmark backend.

Provides dataclasses and functions for running every controller against the
same source and collecting per-window performance stats.  GUI-agnostic — the
Qt layer lives in Source/gui/app.py.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.frame_reader import FrameReader
from ..core.config import RuntimeConfig
from ..core.pipeline import DetectionPipeline
from ..pipelines.pipeline_a import PipelineA
from ..pipelines.pipeline_b import PipelineB
from ..pipelines.pipeline_c import PipelineC
from ..pipelines.pipeline_d import PipelineD
from ..features.extractor import FeatureExtractor
from ..tracking.tracker import TrackerWrapper
from ..evaluation.metrics import WindowMetrics, compute_reward, compute_reward_image
from ..controller.orchestrator import PipelineOrchestrator
from ..controller.base import MetaController
from ..controller.rule_based import RuleBasedController
from ..controller.bandit import UCBBanditController, ContextualBanditController
from ..controller.decision_tree import DecisionTreeController, RandomForestController
from ..controller.neural_net import NeuralNetController
from ..controller.neural_rl import NeuralRLController
from ..controller.none import NoneController

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    controller_name: str
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_switches: int
    switch_rate: float                       # switches / n_windows
    pipeline_distribution: dict[str, float]  # pipeline_name → fraction 0–1
    n_windows: int
    n_frames: int


@dataclass
class PerWindowRecord:
    controller_name: str
    window_index: int
    pipeline_name: str
    reward: float
    mean_latency_ms: float
    mean_confidence: float
    track_consistency: float
    flicker_rate: float


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_pipelines(
    conf: float = 0.30,
    include_heavy: bool = True,
    model_path: str | None = None,
) -> list[DetectionPipeline]:
    """Build detection pipelines. Always includes A, C, D; optionally B (YOLOv8m).

    Parameters
    ----------
    conf         : detection confidence threshold
    include_heavy: whether to include PipelineB (YOLOv8m)
    model_path   : optional path to a custom .pt or .onnx weights file.
                   When None each pipeline uses its own default weights.
    """
    mp_light = model_path or str(_MODELS_DIR / "yolov8n.pt")
    mp_heavy = model_path or str(_MODELS_DIR / "yolov8m.pt")
    pipelines: list[DetectionPipeline] = [
        PipelineA(conf=conf, model_path=mp_light),
        PipelineD(conf=conf, model_path=mp_light),
        PipelineC(conf=conf, model_path=mp_light),
    ]
    if include_heavy:
        try:
            pipelines.append(PipelineB(conf=conf, model_path=mp_heavy))
        except Exception as exc:
            print(f"[WARN] Could not load PipelineB (YOLOv8m): {exc}. Running without it.")
    return pipelines


# ── Controller factory ────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent.parent / "controller" / "models"

_TREE_CONTROLLERS: dict[str, type[DecisionTreeController]] = {
    "decision_tree": DecisionTreeController,
    "random_forest": RandomForestController,
}


def build_controllers(
    selected: list[str],
    pipeline_names: list[str],
    models_dir: Path | None = None,
) -> tuple[list[tuple[str, MetaController]], list[str]]:
    """
    Build controller instances for the given names.

    Returns
    -------
    controllers : list of (name, MetaController)
    warnings    : list of warning strings (e.g. missing model files)
    """
    if models_dir is None:
        models_dir = _MODELS_DIR

    controllers: list[tuple[str, MetaController]] = []
    warnings: list[str] = []

    for name in selected:
        if name == "none":
            controllers.append((name, NoneController()))
        elif name == "rule":
            controllers.append((name, RuleBasedController()))
        elif name == "ucb":
            controllers.append((name, UCBBanditController(pipeline_names)))
        elif name == "contextual":
            controllers.append((name, ContextualBanditController(pipeline_names)))
        elif name in _TREE_CONTROLLERS:
            cls = _TREE_CONTROLLERS[name]
            model_path = models_dir / cls.MODEL_FILENAME
            if not model_path.exists():
                msg = (
                    f"Skipping '{name}': model not found at {model_path}. "
                    f"Run: python -m Source.controller.train_dt"
                )
                warnings.append(msg)
                continue
            controllers.append((name, cls(pipeline_names, model_path=model_path)))
        elif name == "neural_net":
            model_path = models_dir / NeuralNetController.MODEL_FILENAME
            if not model_path.exists():
                warnings.append(
                    f"Skipping 'neural_net': model not found at {model_path}. "
                    f"Run: python -m Source.controller.train_nn"
                )
                continue
            controllers.append((name, NeuralNetController(pipeline_names, model_path=model_path)))
        elif name == "neural_rl":
            model_path = models_dir / NeuralRLController.MODEL_FILENAME
            if not model_path.exists():
                warnings.append(
                    f"Skipping 'neural_rl': model not found at {model_path}. "
                    f"Run: python -m Source.controller.train_rl"
                )
                continue
            controllers.append((name, NeuralRLController(pipeline_names, model_path=model_path)))
        else:
            warnings.append(f"Unknown controller name '{name}' — skipped.")

    return controllers, warnings


# ── Core benchmark loop ───────────────────────────────────────────────────────

def run_controller(
    source_path: str,
    controller: MetaController,
    pipelines: list[DetectionPipeline],
    window_size: int = 30,
    progress_cb: Callable[[str, int], None] | None = None,
) -> tuple[BenchmarkResult, list[PerWindowRecord]]:
    """
    Run one controller through the full source and collect per-window metrics.

    Mirrors ablation.run_video exactly (window boundary, feature extraction
    with empty dets, controller.update() at window end).

    Parameters
    ----------
    source_path  : video file, image, or image directory
    controller   : fresh MetaController instance
    pipelines    : shared (stateless) pipeline objects — safe to reuse
    window_size  : frames per evaluation window
    progress_cb  : optional callback(controller_name, frame_idx) for GUI progress

    Returns
    -------
    (BenchmarkResult, list[PerWindowRecord])
    """
    ctrl_name = type(controller).__name__.replace("Controller", "").lower()
    # Use the controller's own name if it's one of our known short names
    for short in ("rule", "ucb", "contextual", "decision_tree", "random_forest"):
        pass  # name is set by caller via build_controllers — we'll use it below

    reader = FrameReader(source_path)
    extractor = FeatureExtractor()
    tracker = TrackerWrapper()
    window_metrics = WindowMetrics()
    orchestrator = PipelineOrchestrator(
        controller, pipelines, window_size=window_size,
        runtime_config=RuntimeConfig(mode="offline"),
    )
    is_image_mode = reader.source_type in ("image", "directory")

    rewards: list[float] = []
    latencies: list[float] = []
    pipeline_counts: dict[str, int] = {}
    records: list[PerWindowRecord] = []
    n_frames = 0

    for frame in reader:
        n_frames += 1
        if progress_cb:
            progress_cb(ctrl_name, frame.index)

        features = extractor.extract(frame.image, [])
        dets, meta = orchestrator.process(frame, features)
        tracked = tracker.update(dets)
        window_metrics.update(tracked, meta["latency_ms"])
        latencies.append(meta["latency_ms"])

        at_boundary = is_image_mode or ((frame.index + 1) % window_size == 0)
        if at_boundary:
            episode = window_metrics.compute(orchestrator.current_pipeline_name)
            reward = compute_reward_image(episode) if is_image_mode else compute_reward(episode)
            rewards.append(reward)

            # Track pipeline usage
            p_name = orchestrator.current_pipeline_name
            pipeline_counts[p_name] = pipeline_counts.get(p_name, 0) + 1

            records.append(PerWindowRecord(
                controller_name=ctrl_name,
                window_index=len(rewards) - 1,
                pipeline_name=p_name,
                reward=reward,
                mean_latency_ms=episode.latency_ms,
                mean_confidence=episode.mean_confidence,
                track_consistency=episode.track_consistency,
                flicker_rate=episode.flicker_rate,
            ))

            controller.update(
                orchestrator.current_pipeline_name,
                reward,
                orchestrator.feature_buffer[-1] if orchestrator.feature_buffer else None,
            )
            window_metrics.reset()
            if is_image_mode:
                extractor.reset()
                tracker.reset()

    reader.release()
    n_windows = len(rewards)

    if n_windows == 0:
        result = BenchmarkResult(
            controller_name=ctrl_name,
            mean_reward=0.0, std_reward=0.0, min_reward=0.0, max_reward=0.0,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            total_switches=orchestrator.count_switches(),
            switch_rate=0.0,
            pipeline_distribution={},
            n_windows=0,
            n_frames=n_frames,
        )
        return result, records

    r_arr = np.array(rewards)
    dist = {k: v / n_windows for k, v in pipeline_counts.items()}
    result = BenchmarkResult(
        controller_name=ctrl_name,
        mean_reward=float(r_arr.mean()),
        std_reward=float(r_arr.std()),
        min_reward=float(r_arr.min()),
        max_reward=float(r_arr.max()),
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        total_switches=orchestrator.count_switches(),
        switch_rate=orchestrator.count_switches() / n_windows,
        pipeline_distribution=dist,
        n_windows=n_windows,
        n_frames=n_frames,
    )
    return result, records


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(records: list[PerWindowRecord], output_path: str) -> None:
    """Write per-window records to a CSV file."""
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
