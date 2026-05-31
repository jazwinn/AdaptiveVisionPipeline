"""
Per-pipeline and per-controller accuracy evaluation against a labelled
YOLOv8-format dataset (e.g. exported from Roboflow).

Each pipeline's full processing chain (preprocess → infer) is run on every
image in the chosen split and compared against ground-truth YOLO .txt labels
using supervision's MeanAveragePrecision metric.

Controllers are evaluated by letting them choose a pipeline per image (using
single-frame feature extraction) and then running that pipeline.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from ..core.pipeline import DetectionPipeline, Detection
from ..features.extractor import FeatureExtractor

# ── Supervision import (required) ─────────────────────────────────────────────

try:
    import supervision as sv
    from supervision.metrics import MeanAveragePrecision
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_split_files(
    data_yaml: str,
    split: str = "val",
) -> list[tuple[Path, Path]]:
    """
    Parse a YOLOv8 data.yaml and return (image_path, label_path) pairs for
    the requested split.  Images without a matching .txt label file are
    skipped (treated as background — no ground-truth boxes).

    Parameters
    ----------
    data_yaml : path to data.yaml
    split     : "val", "test", or "train"

    Returns
    -------
    list of (image_path, label_path) tuples
    """
    yaml_path = Path(data_yaml).resolve()
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_val = cfg.get(split)
    if split_val is None:
        raise ValueError(
            f"Split '{split}' not found in {yaml_path}. "
            f"Available keys: {list(cfg.keys())}"
        )

    # Resolve the image directory robustly.
    # Roboflow data.yaml path formats vary widely:
    #   "valid/images"          (relative to yaml folder)
    #   "../valid/images"       (relative to yaml's parent folder)
    #   absolute paths
    # We try several candidates and use the first that exists.
    raw      = Path(split_val)
    yaml_dir = yaml_path.parent
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((yaml_dir / raw).resolve())            # yaml folder + raw
        candidates.append((yaml_dir.parent / raw).resolve())     # one level up + raw

    # Folder-name fallbacks: some yamls say "val" but the folder is "valid" (and vice-versa)
    alt_names = [split]
    if split == "val":
        alt_names.append("valid")
    elif split == "valid":
        alt_names.append("val")

    for base in (yaml_dir, yaml_dir.parent):
        for name in alt_names:
            candidates.append((base / name / "images").resolve())

    candidates.append((Path.cwd() / raw).resolve())

    img_dir = next((p for p in candidates if p.exists()), None)
    if img_dir is None:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Image directory for split '{split}' not found. Tried:\n  {tried}"
        )

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        # Derive label path: .../images/... → .../labels/...
        lbl_path = Path(str(img_path).replace(
            str(Path("images")),
            str(Path("labels")),
        )).with_suffix(".txt")
        if not lbl_path.exists():
            # Try sibling labels/ directory
            lbl_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    return pairs


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> "sv.Detections":
    """
    Parse a YOLOv8 label .txt file → supervision Detections (xyxy, class_id).

    YOLO format per line: class_id  cx  cy  w  h  (all normalised 0–1)
    """
    boxes, class_ids = [], []
    try:
        with label_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                boxes.append([x1, y1, x2, y2])
                class_ids.append(cls_id)
    except Exception:
        pass

    if boxes:
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )
    return sv.Detections.empty()


def detections_to_sv(dets: list[Detection]) -> "sv.Detections":
    """Convert pipeline Detection objects → supervision Detections."""
    if not dets:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([d.bbox_xyxy for d in dets], dtype=np.float32),
        confidence=np.array([d.confidence for d in dets], dtype=np.float32),
        class_id=np.array([d.class_id for d in dets], dtype=int),
    )


def _count_tp_fp_fn(
    pred_sv: "sv.Detections",
    gt_sv:   "sv.Detections",
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """
    Count (TP, FP, FN) for one image using greedy IoU matching at ``iou_thr``.

    Each GT box can only be matched once (the highest-IoU prediction wins).
    Used to compute precision and recall across the full split.
    """
    n_pred = len(pred_sv)
    n_gt   = len(gt_sv)

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    iou_matrix = sv.box_iou_batch(pred_sv.xyxy, gt_sv.xyxy)  # (n_pred, n_gt)
    matched_gt: set[int] = set()
    tp = 0
    for pred_i in range(n_pred):
        best_j = int(iou_matrix[pred_i].argmax())
        if iou_matrix[pred_i, best_j] >= iou_thr and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)

    fp = n_pred - tp
    fn = n_gt   - len(matched_gt)
    return tp, fp, fn


# ── Metric extraction helpers ──────────────────────────────────────────────────

def _extract_map_metrics(result) -> tuple[float, float, float, float]:
    """
    Extract (map50, map50_95, mean_precision, mean_recall) from a
    supervision MeanAveragePrecision result object.
    Handles slight API differences across supervision versions.
    """
    map50    = float(getattr(result, "map50",    0.0))
    map50_95 = float(getattr(result, "map50_95", getattr(result, "map", 0.0)))

    # precision / recall not always on the result; default 0
    mp = float(getattr(result, "mean_precision", getattr(result, "mp", 0.0)))
    mr = float(getattr(result, "mean_recall",    getattr(result, "mr", 0.0)))
    return map50, map50_95, mp, mr


# ── Per-pipeline evaluation ────────────────────────────────────────────────────

def eval_one_pipeline(
    pipeline: DetectionPipeline,
    file_pairs: list[tuple[Path, Path]],
    conf: float = 0.25,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Evaluate a single pipeline on all (image, label) pairs.

    Returns a dict with pipeline_name, map50, map, precision, recall,
    mean_latency_ms.
    """
    if not _SV_AVAILABLE:
        raise RuntimeError("supervision is required for pipeline evaluation. pip install supervision")

    metric = MeanAveragePrecision()
    latencies: list[float] = []
    total_tp = total_fp = total_fn = 0
    n = len(file_pairs)

    for i, (img_path, lbl_path) in enumerate(file_pairs):
        if progress_cb:
            progress_cb(pipeline.name, i, n)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        preprocessed = pipeline.preprocess(img)
        dets = pipeline.infer(preprocessed)
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_sv = detections_to_sv([d for d in dets if d.confidence >= conf])
        gt_sv   = load_ground_truth(lbl_path, w, h)
        metric.update(predictions=[pred_sv], targets=[gt_sv])

        tp, fp, fn = _count_tp_fp_fn(pred_sv, gt_sv)
        total_tp += tp; total_fp += fp; total_fn += fn

    result = metric.compute()
    map50, map50_95, _, _ = _extract_map_metrics(result)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "pipeline_name":   pipeline.name,
        "map50":           map50,
        "map":             map50_95,
        "precision":       precision,
        "recall":          recall,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


# ── Per-controller evaluation ──────────────────────────────────────────────────

def eval_one_controller(
    controller,
    pipelines: list[DetectionPipeline],
    file_pairs: list[tuple[Path, Path]],
    conf: float = 0.25,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Evaluate a controller by letting it choose a pipeline per image using
    single-frame feature extraction, then running that pipeline against GT.

    Returns a dict with controller_name, map50, map, precision, recall,
    mean_latency_ms, pipeline_distribution.
    """
    if not _SV_AVAILABLE:
        raise RuntimeError("supervision is required for controller evaluation.")

    extractor    = FeatureExtractor()
    pipeline_map = {p.name: p for p in pipelines}
    pipeline_names = [p.name for p in pipelines]
    ctrl_name    = getattr(controller, "__class__", type(controller)).__name__

    metric = MeanAveragePrecision()
    latencies: list[float] = []
    pipeline_counts: dict[str, int] = {}
    total_tp = total_fp = total_fn = 0
    n = len(file_pairs)

    for i, (img_path, lbl_path) in enumerate(file_pairs):
        if progress_cb:
            progress_cb(ctrl_name, i, n)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ask controller which pipeline to use for this image
        features = extractor.extract(img, [])
        chosen   = controller.select_pipeline([features], pipeline_names)
        pipeline = pipeline_map.get(chosen, pipelines[0])
        pipeline_counts[pipeline.name] = pipeline_counts.get(pipeline.name, 0) + 1

        t0 = time.perf_counter()
        preprocessed = pipeline.preprocess(img)
        dets = pipeline.infer(preprocessed)
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_sv = detections_to_sv([d for d in dets if d.confidence >= conf])
        gt_sv   = load_ground_truth(lbl_path, w, h)
        metric.update(predictions=[pred_sv], targets=[gt_sv])

        tp, fp, fn = _count_tp_fp_fn(pred_sv, gt_sv)
        total_tp += tp; total_fp += fp; total_fn += fn

    result = metric.compute()
    map50, map50_95, _, _ = _extract_map_metrics(result)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    total = max(len(file_pairs), 1)
    dist  = {k: v / total for k, v in pipeline_counts.items()}

    return {
        "controller_name":       ctrl_name,
        "map50":                 map50,
        "map":                   map50_95,
        "precision":             precision,
        "recall":                recall,
        "mean_latency_ms":       float(np.mean(latencies)) if latencies else 0.0,
        "pipeline_distribution": dist,
    }


# ── Public comparison runners ──────────────────────────────────────────────────

def run_pipeline_comparison(
    data_yaml: str,
    split: str,
    conf: float,
    model_path: str | None,
    selected_pipelines: list[str],
    progress_cb: Callable[[str, int, int], None] | None = None,
    result_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Evaluate each selected pipeline and return a list of result dicts.

    ``result_cb`` is called immediately after each pipeline finishes so the
    GUI can add a row without waiting for the full batch to complete.
    """
    from .benchmark import build_pipelines

    file_pairs = load_split_files(data_yaml, split)
    if not file_pairs:
        raise FileNotFoundError(f"No labelled images found in split '{split}' of {data_yaml}")

    all_pipelines = build_pipelines(conf=conf, include_heavy=True, model_path=model_path)
    pipeline_map  = {p.name: p for p in all_pipelines}

    results: list[dict] = []
    for name in selected_pipelines:
        pipeline = pipeline_map.get(name)
        if pipeline is None:
            warnings.warn(f"Pipeline '{name}' not found — skipped.")
            continue
        r = eval_one_pipeline(pipeline, file_pairs, conf=conf, progress_cb=progress_cb)
        results.append(r)
        if result_cb:
            result_cb(r)

    return results


def run_controller_comparison(
    data_yaml: str,
    split: str,
    conf: float,
    model_path: str | None,
    selected_controllers: list[str],
    progress_cb: Callable[[str, int, int], None] | None = None,
    result_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Evaluate each selected controller and return a list of result dicts.

    ``result_cb`` is called immediately after each controller finishes so the
    GUI can add a row without waiting for the full batch to complete.
    """
    from .benchmark import build_pipelines, build_controllers

    file_pairs = load_split_files(data_yaml, split)
    if not file_pairs:
        raise FileNotFoundError(f"No labelled images found in split '{split}' of {data_yaml}")

    pipelines     = build_pipelines(conf=conf, include_heavy=True, model_path=model_path)
    pipeline_names = [p.name for p in pipelines]
    controllers, warns = build_controllers(selected_controllers, pipeline_names)
    for w in warns:
        warnings.warn(w)

    results: list[dict] = []
    for ctrl_idx, (ctrl_label, ctrl) in enumerate(controllers):
        # Wrap the callback so the GUI receives the short label (e.g. "rule") and
        # the global controller index — not the raw class name from eval_one_controller.
        if progress_cb:
            def _make_cb(label: str, idx: int) -> Callable[[str, int, int], None]:
                def _cb(_name: str, i: int, n: int) -> None:
                    progress_cb(label, i, n)
                return _cb
            wrapped_cb: Callable[[str, int, int], None] | None = _make_cb(ctrl_label, ctrl_idx)
        else:
            wrapped_cb = None

        r = eval_one_controller(ctrl, pipelines, file_pairs, conf=conf, progress_cb=wrapped_cb)
        r["controller_name"] = ctrl_label
        results.append(r)
        if result_cb:
            result_cb(r)

    return results
