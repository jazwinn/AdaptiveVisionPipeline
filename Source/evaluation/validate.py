"""
Model validation against a labelled YOLOv8-format dataset.

Wraps ultralytics YOLO.val() and returns a structured dict of metrics.
Can be imported by the GUI or called from the CLI.

Usage (CLI)::

    python -m Source.evaluation.validate \\
        --model  models/yolov8n.pt \\
        --data   /path/to/dataset/data.yaml \\
        --split  val \\
        --conf   0.25 \\
        --imgsz  640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_validation(
    model_path: str,
    data_yaml: str,
    split: str = "val",     # "val" or "test" (matches key in data.yaml)
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict:
    """
    Run YOLO validation and return a structured metrics dict.

    Parameters
    ----------
    model_path : path to .pt or .onnx weights file
    data_yaml  : path to Roboflow / YOLOv8 data.yaml
    split      : dataset split to evaluate — "val", "test", or "train"
    conf       : confidence threshold for predictions
    imgsz      : inference image size

    Returns
    -------
    dict with keys:
        map50, map, precision, recall,
        per_class (list of dicts),
        speed (dict),
        n_classes (int)
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    # Per-class breakdown
    per_class: list[dict] = []
    names: dict[int, str] = results.names

    ap_class_index = results.box.ap_class_index
    ap50_arr       = results.box.ap50
    ap_arr         = results.box.ap
    # p and r are per-class arrays; mp/mr are their means
    p_arr = getattr(results.box, "p", None)
    r_arr = getattr(results.box, "r", None)

    for i, cls_idx in enumerate(ap_class_index):
        per_class.append({
            "class":     names.get(int(cls_idx), str(cls_idx)),
            "ap50":      float(ap50_arr[i]),
            "ap":        float(ap_arr[i]),
            "precision": float(p_arr[i]) if p_arr is not None and i < len(p_arr) else 0.0,
            "recall":    float(r_arr[i]) if r_arr is not None and i < len(r_arr) else 0.0,
        })

    return {
        "map50":     float(results.box.map50),
        "map":       float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "per_class": per_class,
        "speed":     results.speed,   # {"preprocess": ms, "inference": ms, "postprocess": ms}
        "n_classes": len(per_class),
        "split":     split,
        "model":     str(model_path),
        "data":      str(data_yaml),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a YOLO model against a labelled YOLOv8 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  required=True,            help="Path to .pt / .onnx weights")
    p.add_argument("--data",   required=True,            help="Path to data.yaml")
    p.add_argument("--split",  default="val",            help="Dataset split: val / test / train")
    p.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz",  type=int,   default=640,  help="Inference image size")
    p.add_argument("--output", default=None,             help="Optional path to save JSON report")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\nValidating: {args.model}")
    print(f"Dataset:    {args.data}  (split={args.split})")
    print(f"Conf={args.conf}  imgsz={args.imgsz}\n")

    metrics = run_validation(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    print(f"mAP50:     {metrics['map50']:.4f}  ({metrics['map50']*100:.1f}%)")
    print(f"mAP50-95:  {metrics['map']:.4f}  ({metrics['map']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    spd = metrics["speed"]
    print(f"Speed:     pre={spd.get('preprocess', 0):.1f}ms  "
          f"infer={spd.get('inference', 0):.1f}ms  "
          f"post={spd.get('postprocess', 0):.1f}ms")

    if metrics["per_class"]:
        print(f"\nPer-class ({metrics['n_classes']} classes):")
        print(f"  {'Class':<20} {'AP50':>6}  {'AP':>6}  {'P':>6}  {'R':>6}")
        for cls in metrics["per_class"]:
            print(f"  {cls['class']:<20} {cls['ap50']:>6.3f}  {cls['ap']:>6.3f}"
                  f"  {cls['precision']:>6.3f}  {cls['recall']:>6.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
