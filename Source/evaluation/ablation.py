from __future__ import annotations
from typing import Callable
import numpy as np
from ..core.frame_reader import FrameReader
from ..core.pipeline import DetectionPipeline
from ..features.extractor import FeatureExtractor
from ..tracking.tracker import TrackerWrapper
from ..evaluation.metrics import WindowMetrics, compute_reward
from ..controller.orchestrator import PipelineOrchestrator
from ..controller.base import MetaController


def run_video(
    video_path: str,
    orchestrator: PipelineOrchestrator,
    max_frames: int | None = None,
) -> list[float]:
    reader = FrameReader(video_path)
    extractor = FeatureExtractor()
    tracker = TrackerWrapper()
    window_metrics = WindowMetrics()
    rewards: list[float] = []
    window_size = orchestrator.window_size

    for frame in reader:
        if max_frames and frame.index >= max_frames:
            break

        # Run with empty dets for first frame feature extraction
        dets, meta = orchestrator.process(frame, extractor.extract(frame.image, []))
        tracked = tracker.update(dets)
        window_metrics.update(tracked, meta["latency_ms"])

        # At window boundary, compute reward and reset
        if (frame.index + 1) % window_size == 0:
            episode = window_metrics.compute(orchestrator.current_pipeline_name)
            reward = compute_reward(episode)
            rewards.append(reward)
            orchestrator.controller.update(
                orchestrator.current_pipeline_name,
                reward,
                orchestrator.feature_buffer[-1] if orchestrator.feature_buffer else None,
            )
            window_metrics.reset()

    reader.release()
    return rewards


def ablate_window_size(
    video_path: str,
    controller_factory: Callable[[], MetaController],
    pipelines: list[DetectionPipeline],
    window_sizes: list[int] | None = None,
    max_frames: int = 500,
) -> dict[int, dict]:
    if window_sizes is None:
        window_sizes = [10, 30, 60, 120]

    results = {}
    for ws in window_sizes:
        controller = controller_factory()
        orchestrator = PipelineOrchestrator(controller, pipelines, window_size=ws)
        rewards = run_video(video_path, orchestrator, max_frames=max_frames)
        results[ws] = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "pipeline_switches": orchestrator.count_switches(),
            "n_windows": len(rewards),
        }
    return results


def ablate_conf_threshold(
    video_path: str,
    pipeline_factory: Callable[[float], DetectionPipeline],
    controller_factory: Callable[[], MetaController],
    thresholds: list[float] | None = None,
    max_frames: int = 500,
) -> dict[float, dict]:
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5]

    results = {}
    for thresh in thresholds:
        pipeline = pipeline_factory(thresh)
        controller = controller_factory()
        orchestrator = PipelineOrchestrator(controller, [pipeline], window_size=30)
        rewards = run_video(video_path, orchestrator, max_frames=max_frames)
        results[thresh] = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
        }
    return results
