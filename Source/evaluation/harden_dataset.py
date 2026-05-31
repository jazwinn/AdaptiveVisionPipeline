"""
harden_dataset.py
-----------------
Creates degraded copies of a YOLOv8 split so fast_baseline drops to ~60-75%
mAP50, making CLAHE's advantage visible to the adaptive controller.

Labels are copied unchanged — bounding boxes are still valid after image-level
degradation. Originals are never modified.

Usage:
    python Source/evaluation/harden_dataset.py --dataset-dir "Solar Panel.v4i.yolov8" --split test --difficulty hard
    python Source/evaluation/harden_dataset.py --dataset-dir "Solar Panel.v4i.yolov8" --split test valid --difficulty medium
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


# ── Degradation functions ──────────────────────────────────────────────────────

def apply_brightness_reduction(img: np.ndarray, factor: float = 0.55) -> np.ndarray:
    """Multiply pixel values by factor to simulate shadows / underexposure."""
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray, ksize: int = 7, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur to simulate motion blur or out-of-focus optics."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """Additive Gaussian noise to simulate sensor noise."""
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    return (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def apply_haze_overlay(
    img: np.ndarray,
    intensity: float = 0.40,
    haze_color: tuple[int, int, int] = (220, 220, 200),
) -> np.ndarray:
    """Alpha-blend a flat haze layer to simulate atmospheric fog."""
    haze = np.full_like(img, fill_value=haze_color, dtype=np.uint8)
    return cv2.addWeighted(img, 1.0 - intensity, haze, intensity, 0)


def apply_low_contrast(img: np.ndarray, alpha: float = 0.60, beta: float = 50.0) -> np.ndarray:
    """Compress tonal range so panels blend into dark rooftops (out = alpha*img + beta)."""
    return (img.astype(np.float32) * alpha + beta).clip(0, 255).astype(np.uint8)


def apply_overexposure(img: np.ndarray, factor: float = 1.9) -> np.ndarray:
    """Multiply and clip to simulate glare or reflective panel overexposure."""
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


# ── Difficulty presets ─────────────────────────────────────────────────────────

# (function, kwargs, probability_of_inclusion_per_image)
DegradationSpec = tuple[Callable, dict, float]

DIFFICULTY_CONFIGS: dict[str, list[DegradationSpec]] = {
    "easy": [
        (apply_brightness_reduction, {"factor": 0.75},              0.30),
        (apply_gaussian_blur,        {"ksize": 3, "sigma": 1.0},    0.30),
        (apply_gaussian_noise,       {"mean": 0.0, "std": 12.0},    0.30),
        (apply_haze_overlay,         {"intensity": 0.20},            0.20),
        (apply_low_contrast,         {"alpha": 0.75, "beta": 30.0}, 0.25),
        (apply_overexposure,         {"factor": 1.4},                0.20),
    ],
    "medium": [
        (apply_brightness_reduction, {"factor": 0.55},              0.50),
        (apply_gaussian_blur,        {"ksize": 7, "sigma": 2.0},    0.45),
        (apply_gaussian_noise,       {"mean": 0.0, "std": 25.0},    0.45),
        (apply_haze_overlay,         {"intensity": 0.35},            0.40),
        (apply_low_contrast,         {"alpha": 0.60, "beta": 50.0}, 0.45),
        (apply_overexposure,         {"factor": 1.9},                0.30),
    ],
    "hard": [
        (apply_brightness_reduction, {"factor": 0.35},              0.70),
        (apply_gaussian_blur,        {"ksize": 11, "sigma": 3.5},   0.65),
        (apply_gaussian_noise,       {"mean": 0.0, "std": 45.0},    0.60),
        (apply_haze_overlay,         {"intensity": 0.55},            0.55),
        (apply_low_contrast,         {"alpha": 0.45, "beta": 70.0}, 0.60),
        (apply_overexposure,         {"factor": 2.4},                0.40),
    ],
}


# ── Core logic ─────────────────────────────────────────────────────────────────

def degrade_image(
    img: np.ndarray,
    specs: list[DegradationSpec],
    rng: np.random.RandomState,
    min_transforms: int = 1,
    max_transforms: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of degradation functions to one image."""
    selected: list[DegradationSpec] = []
    remaining: list[DegradationSpec] = []

    for spec in specs:
        fn, kwargs, prob = spec
        if rng.random() < prob:
            selected.append(spec)
        else:
            remaining.append(spec)

    # Enforce max
    if len(selected) > max_transforms:
        indices = sorted(rng.choice(len(selected), max_transforms, replace=False).tolist())
        selected = [selected[i] for i in indices]

    # Enforce min
    while len(selected) < min_transforms and remaining:
        idx = int(rng.randint(0, len(remaining)))
        selected.append(remaining.pop(idx))

    out = img.copy()
    applied: list[str] = []
    for fn, kwargs, _ in selected:
        out = fn(out, **kwargs)
        applied.append(fn.__name__)

    return out, applied


def setup_labels_dir(src: Path, dst: Path, use_symlink: bool = False) -> None:
    """Copy (or symlink) the labels directory to dst, replacing any prior copy."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copytree(src, dst)


def harden_split(
    split: str,
    difficulty: str,
    dataset_dir: Path,
    seed: int = SEED,
    use_symlink: bool = False,
) -> dict:
    """Process one split and return a summary dict."""
    specs = DIFFICULTY_CONFIGS[difficulty]
    rng   = np.random.RandomState(seed)

    src_images = dataset_dir / split / "images"
    src_labels = dataset_dir / split / "labels"
    out_dir    = dataset_dir / f"{split}_hard"
    dst_images = out_dir / "images"
    dst_labels = out_dir / "labels"

    if not src_images.exists():
        raise FileNotFoundError(f"Source images not found: {src_images}")
    if not src_labels.exists():
        raise FileNotFoundError(f"Source labels not found: {src_labels}")

    dst_images.mkdir(parents=True, exist_ok=True)
    setup_labels_dir(src_labels, dst_labels, use_symlink=use_symlink)

    img_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS)

    n_written = 0
    n_failed  = 0
    transform_counts: dict[str, int] = {}

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            n_failed += 1
            continue

        degraded, applied = degrade_image(img, specs, rng)

        dst_path = dst_images / img_path.name
        if cv2.imwrite(str(dst_path), degraded):
            n_written += 1
            for name in applied:
                transform_counts[name] = transform_counts.get(name, 0) + 1
        else:
            n_failed += 1

    return {
        "split":            split,
        "difficulty":       difficulty,
        "n_images":         len(img_paths),
        "n_written":        n_written,
        "n_failed":         n_failed,
        "transform_counts": transform_counts,
        "output_dir":       str(out_dir),
    }


def write_data_hard_yaml(dataset_dir: Path, processed_splits: list[str]) -> Path:
    """Write data_hard.yaml pointing hard splits at *_hard variants.

    A split is pointed at its _hard folder if it was in ``processed_splits``
    OR if the ``<split>_hard/images`` folder already exists on disk (so
    running ``--split valid`` after a previous ``--split test`` run doesn't
    silently revert the test entry back to the original path).
    """
    def _hard_if_exists(split: str, orig_name: str) -> str:
        hard_dir = dataset_dir / f"{split}_hard" / "images"
        if split in processed_splits or hard_dir.exists():
            # No "../" — the _hard folders are siblings of data_hard.yaml inside
            # the dataset directory, so a plain relative path resolves correctly.
            return f"{split}_hard/images"
        return f"../{orig_name}/images"

    train_path = "../train/images"
    valid_path = _hard_if_exists("valid", "valid")
    test_path  = _hard_if_exists("test",  "test")

    content = (
        f"train: {train_path}\n"
        f"val:   {valid_path}\n"
        f"test:  {test_path}\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['panel']\n"
        f"\n"
        f"# Generated by harden_dataset.py — train split is original (not degraded).\n"
    )

    out_path = dataset_dir / "data_hard.yaml"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def print_summary(summaries: list[dict], yaml_path: Path) -> None:
    print("\n" + "=" * 60)
    print("  HARDEN DATASET SUMMARY")
    print("=" * 60)
    for s in summaries:
        print(f"\nSplit:      {s['split']}  ->  {s['split']}_hard")
        print(f"Difficulty: {s['difficulty']}")
        print(f"Images:     {s['n_written']} written / {s['n_images']} source "
              f"({s['n_failed']} failed)")
        print(f"Output dir: {s['output_dir']}")
        print("Transforms applied (# images each):")
        if s["transform_counts"]:
            for fn_name, count in sorted(s["transform_counts"].items()):
                pct = count / max(s["n_written"], 1) * 100
                print(f"  {fn_name:<35} {count:>5}  ({pct:.1f}%)")
        else:
            print("  (none)")
    print(f"\ndata_hard.yaml written to: {yaml_path}")
    print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create harder degraded copies of YOLOv8 dataset splits.\n"
            "Originals are never modified. Labels are copied unchanged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python Source/evaluation/harden_dataset.py --split test --difficulty hard\n"
            "  python Source/evaluation/harden_dataset.py --split test valid --difficulty medium\n"
        ),
    )
    p.add_argument(
        "--split", nargs="+", default=["test"],
        choices=["train", "valid", "test"],
        help="Which split(s) to harden.",
    )
    p.add_argument(
        "--difficulty", default="hard",
        choices=["easy", "medium", "hard"],
        help="Degradation intensity level.",
    )
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for reproducibility.")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink labels dir instead of copying (requires admin on Windows).")
    p.add_argument("--dataset-dir", required=True,
                   help="Path to the dataset root (contains train/, valid/, test/, data.yaml).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ddir = Path(args.dataset_dir)

    if not ddir.exists():
        raise SystemExit(f"Dataset directory not found: {ddir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    summaries: list[dict] = []
    for split in args.split:
        print(f"[harden] Processing split='{split}', difficulty='{args.difficulty}' ...")
        s = harden_split(
            split=split,
            difficulty=args.difficulty,
            dataset_dir=ddir,
            seed=args.seed,
            use_symlink=args.symlink,
        )
        summaries.append(s)

    yaml_path = write_data_hard_yaml(ddir, processed_splits=args.split)
    print_summary(summaries, yaml_path)


if __name__ == "__main__":
    main()
