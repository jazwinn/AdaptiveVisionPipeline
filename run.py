"""
Main entrypoint for the adaptive drone detection pipeline.

Usage:
    python run.py --source drone_footage.mp4
    python run.py --source photo.jpg --display
    python run.py --source frames/ --output annotated_frames/
    python run.py --source footage.mp4 --controller ucb --window 30 --mode offline
    python run.py --source footage.mp4 --ablate window
"""
from __future__ import annotations
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# Some OpenCV builds (headless) lack GUI support; detect once at startup.
def _check_display() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False

_HAS_DISPLAY = _check_display()
from rich.console import Console

from drone_detect.core.config import RunConfig, RuntimeConfig
from drone_detect.core.frame_reader import FrameReader, _IMAGE_EXTS
from drone_detect.core.pipeline import Detection
from drone_detect.pipelines.pipeline_a import PipelineA
from drone_detect.pipelines.pipeline_b import PipelineB
from drone_detect.pipelines.pipeline_c import PipelineC
from drone_detect.pipelines.pipeline_d import PipelineD
from drone_detect.features.extractor import FeatureExtractor
from drone_detect.tracking.tracker import TrackerWrapper
from drone_detect.controller.rule_based import RuleBasedController
from drone_detect.controller.bandit import UCBBanditController, ContextualBanditController
from drone_detect.controller.orchestrator import PipelineOrchestrator
from drone_detect.evaluation.metrics import WindowMetrics, compute_reward
from drone_detect.evaluation.replay_buffer import ReplayBuffer
from drone_detect.experiments.logger import ExperimentLogger

console = Console()


def build_pipelines(args) -> list:
    pipelines = [PipelineA(conf=args.conf)]
    if not args.fast_only:
        pipelines.append(PipelineD(conf=args.conf))
        pipelines.append(PipelineC(conf=args.conf))
        if args.heavy:
            pipelines.append(PipelineB(conf=args.conf))
    return pipelines


def build_controller(name: str, pipeline_names: list[str]):
    if name == "rule":
        return RuleBasedController()
    elif name == "ucb":
        return UCBBanditController(pipeline_names)
    elif name == "contextual":
        return ContextualBanditController(pipeline_names)
    else:
        console.print(f"[red]Unknown controller '{name}', defaulting to rule-based[/red]")
        return RuleBasedController()


def dets_to_sv(dets: list[Detection]) -> sv.Detections:
    if not dets:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([d.bbox_xyxy for d in dets]),
        confidence=np.array([d.confidence for d in dets]),
        class_id=np.array([d.class_id for d in dets]),
    )


def _output_is_video(output_path: str) -> bool:
    return Path(output_path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def run_main(args):
    source_path = args.source
    if not Path(source_path).exists():
        console.print(f"[red]Source not found: {source_path}[/red]")
        sys.exit(1)

    pipelines = build_pipelines(args)
    pipeline_names = [p.name for p in pipelines]
    controller = build_controller(args.controller, pipeline_names)

    # Guard: realtime mode is meaningless for still images
    mode = args.mode
    reader = FrameReader(source_path, max_fps=args.target_fps if mode == "realtime" else None)
    if reader.source_type in ("image", "directory") and mode == "realtime":
        console.print("[yellow]Warning: --mode realtime ignored for image/directory sources.[/yellow]")
        mode = "offline"

    runtime_cfg = RuntimeConfig(mode=mode, target_fps=args.target_fps, max_pipeline_cost=args.max_cost)
    orchestrator = PipelineOrchestrator(controller, pipelines, window_size=args.window, runtime_config=runtime_cfg)

    run_config = RunConfig(
        source_path=source_path,
        controller_type=args.controller,
        pipeline_names=pipeline_names,
        window_size=args.window,
        notes=args.notes,
    )
    logger = ExperimentLogger(run_config) if args.log else None
    replay = ReplayBuffer() if args.replay else None
    extractor = FeatureExtractor()
    tracker = TrackerWrapper()
    window_metrics = WindowMetrics()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Console header
    source_label = {
        "video": "Video",
        "image": "Image",
        "directory": f"Directory ({reader.frame_count} images)",
    }[reader.source_type]
    console.print(f"[bold]{source_label}:[/bold] {source_path}  |  {reader.width}x{reader.height} @ {reader.fps:.1f} fps")
    console.print(f"[bold]Pipelines:[/bold] {pipeline_names}")
    console.print(f"[bold]Controller:[/bold] {args.controller}  |  window={args.window}  |  mode={mode}")
    console.rule()

    if args.display and not _HAS_DISPLAY:
        console.print("[yellow]Warning: --display requested but OpenCV has no GUI support (headless build). "
                      "Install opencv-python (not opencv-python-headless) to enable preview.[/yellow]")

    last_reward = 0.0
    pipeline_counts: dict[str, int] = {n: 0 for n in pipeline_names}
    latency_history: deque[float] = deque(maxlen=30)

    # Output setup
    out_writer = None
    out_dir: Path | None = None
    out_image_path: str | None = None
    last_annotated: np.ndarray | None = None

    if args.output:
        if reader.source_type == "image":
            out_image_path = args.output
        elif reader.source_type == "directory":
            if _output_is_video(args.output):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(args.output, fourcc, reader.fps, (reader.width, reader.height))
            else:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
        else:  # video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.output, fourcc, reader.fps, (reader.width, reader.height))

    try:
        for frame in reader:
            if args.max_frames and frame.index >= args.max_frames:
                break

            features = extractor.extract(frame.image, [])
            dets, meta = orchestrator.process(frame, features)
            tracked = tracker.update(dets)
            window_metrics.update(tracked, meta["latency_ms"])
            pipeline_counts[meta["selected_pipeline"]] = (
                pipeline_counts.get(meta["selected_pipeline"], 0) + 1
            )
            latency_history.append(meta["latency_ms"])

            # Window boundary: compute reward, update controller
            if (frame.index + 1) % args.window == 0:
                episode = window_metrics.compute(orchestrator.current_pipeline_name)
                last_reward = compute_reward(episode)
                features_snap = orchestrator.feature_buffer[-1] if orchestrator.feature_buffer else None
                controller.update(orchestrator.current_pipeline_name, last_reward, features_snap)
                if replay and features_snap:
                    replay.append(features_snap, orchestrator.current_pipeline_name, last_reward)
                window_metrics.reset()

            if logger:
                logger.log_frame(frame.index, meta["selected_pipeline"], features, dets, last_reward, meta["latency_ms"])

            # Annotate
            sv_dets = dets_to_sv(dets)
            annotated = frame.image.copy()
            if len(sv_dets) > 0:
                labels = [f"#{t.track_id} {t.class_name} {t.confidence:.2f}" for t in tracked]
                annotated = box_annotator.annotate(annotated, sv_dets)
                annotated = label_annotator.annotate(annotated, sv_dets, labels=labels)
            cv2.putText(
                annotated,
                f"{meta['selected_pipeline']} | {meta['latency_ms']:.0f}ms | r={last_reward:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            last_annotated = annotated

            # Write output
            if out_writer:
                out_writer.write(annotated)
            elif out_dir:
                ext = Path(frame.source_path).suffix or ".jpg"
                cv2.imwrite(str(out_dir / f"{frame.index:06d}{ext}"), annotated)

            # Display
            if args.display and _HAS_DISPLAY:
                cv2.imshow("Adaptive Detection", annotated)
                wait_ms = 0 if reader.source_type == "image" else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break

            # Console status (every 30 frames, always for single images)
            if frame.index % 30 == 0 or reader.source_type == "image":
                avg_lat = np.mean(latency_history) if latency_history else 0
                console.print(
                    f"[dim]Frame {frame.index:5d}[/dim]  "
                    f"pipeline=[bold cyan]{meta['selected_pipeline']}[/bold cyan]  "
                    f"dets={len(dets):2d}  "
                    f"lat={meta['latency_ms']:5.1f}ms  "
                    f"avg={avg_lat:5.1f}ms  "
                    f"reward={last_reward:+.3f}"
                )

    finally:
        reader.release()
        if out_writer:
            out_writer.release()
        # Save single image output at the end
        if out_image_path and last_annotated is not None:
            cv2.imwrite(out_image_path, last_annotated)
            console.print(f"Annotated image saved to [green]{out_image_path}[/green]")
        if _HAS_DISPLAY:
            cv2.destroyAllWindows()

    if logger:
        csv_path = logger.save()
        console.rule("[bold]Run complete[/bold]")
        console.print(f"Results saved to [green]{csv_path}[/green]")
        console.print(logger.summary())

    console.print(f"\n[bold]Pipeline usage:[/bold] {pipeline_counts}")
    console.print(f"[bold]Total switches:[/bold] {orchestrator.count_switches()}")


def run_ablation(args):
    from drone_detect.evaluation.ablation import ablate_window_size
    console.print(f"[bold]Ablation: {args.ablate}[/bold]")

    pipelines = build_pipelines(args)

    if args.ablate == "window":
        results = ablate_window_size(
            args.source,
            lambda: build_controller(args.controller, [p.name for p in pipelines]),
            pipelines,
            max_frames=args.max_frames or 300,
        )
        console.print("\n[bold]Window size ablation:[/bold]")
        for ws, r in results.items():
            console.print(
                f"  window={ws:4d}  mean_reward={r['mean_reward']:+.4f}  "
                f"std={r['reward_std']:.4f}  switches={r['pipeline_switches']}"
            )
    else:
        console.print(f"[red]Unknown ablation target: {args.ablate}[/red]")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive drone detection pipeline")
    parser.add_argument("--source", required=True,
                        help="Input: video file, image file (.jpg/.png/etc.), or directory of images")
    parser.add_argument("--controller", default="rule", choices=["rule", "ucb", "contextual"])
    parser.add_argument("--window", type=int, default=30, help="Controller decision window (frames)")
    parser.add_argument("--mode", default="offline", choices=["offline", "realtime"])
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-cost", type=float, default=999.0, help="Max pipeline cost in realtime mode")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--display", action="store_true", help="Show preview window")
    parser.add_argument("--output", default=None,
                        help="Output path: image file, directory (for image inputs), or .mp4")
    parser.add_argument("--log", action="store_true", help="Save per-frame CSV log")
    parser.add_argument("--replay", action="store_true", help="Append to replay buffer")
    parser.add_argument("--fast-only", action="store_true", help="Use only fast_baseline pipeline")
    parser.add_argument("--heavy", action="store_true", help="Include yolov8m high-res pipeline")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--ablate", default=None, choices=["window", "conf"],
                        help="Run ablation study instead of normal run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ablate:
        run_ablation(args)
    else:
        run_main(args)
