"""
Train Decision Tree / Random Forest meta-controllers from replay buffer data.

Usage (from any working directory):
    python train_dt.py [--replay PATH] [--model dt|rf|both] [--output DIR] [--top-k-features N]

If no replay buffer file is found, 500 synthetic samples are generated automatically
using rule-based heuristics so training works out-of-the-box without real data.

Outputs:
    dt_controller.joblib   -- DecisionTreeClassifier artifact
    rf_controller.joblib   -- RandomForestClassifier artifact
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURE_NAMES: list[str] = [
    "laplacian_variance",
    "fft_blur_score",
    "mean_intensity",
    "intensity_std",
    "underexposed_ratio",
    "overexposed_ratio",
    "optical_flow_magnitude",
    "frame_displacement",
    "mean_confidence",
    "detection_count",
    "small_object_ratio",
    "edge_density",
    "entropy",
]
AGG_FEATURE_NAMES: list[str] = (
    [f"mean_{f}" for f in FEATURE_NAMES] + [f"std_{f}" for f in FEATURE_NAMES]
)

PIPELINE_NAMES = ["fast_baseline", "clahe_pipeline", "tiled", "high_res"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_replay_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def generate_synthetic_data(n_samples: int = 500) -> list[dict]:
    """Synthetic samples using rule-based heuristics + random noise."""
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n_samples):
        feat = {
            "laplacian_variance": float(rng.uniform(10, 1500)),
            "fft_blur_score":     float(rng.uniform(0, 100)),
            "mean_intensity":     float(rng.uniform(20, 235)),
            "intensity_std":      float(rng.uniform(5, 80)),
            "underexposed_ratio": float(rng.uniform(0, 0.5)),
            "overexposed_ratio":  float(rng.uniform(0, 0.3)),
            "optical_flow_magnitude": float(rng.uniform(0, 20)),
            "frame_displacement": float(rng.uniform(0, 30)),
            "mean_confidence":    float(rng.uniform(0.1, 0.95)),
            "detection_count":    int(rng.integers(0, 30)),
            "small_object_ratio": float(rng.uniform(0, 1)),
            "edge_density":       float(rng.uniform(0, 0.4)),
            "entropy":            float(rng.uniform(2, 8)),
        }
        # Mirror rule_based.py heuristics for labels
        if feat["mean_intensity"] < 60 and feat["intensity_std"] < 25:
            pipeline = "clahe_pipeline"
        elif feat["optical_flow_magnitude"] > 8.0:
            pipeline = "fast_baseline"
        elif feat["small_object_ratio"] > 0.5 or feat["edge_density"] > 0.15:
            pipeline = "tiled"
        elif feat["mean_confidence"] < 0.35 and feat["detection_count"] < 3:
            pipeline = "high_res"
        else:
            pipeline = "fast_baseline"

        reward = float(rng.uniform(-0.5, 2.5))
        records.append({"features": feat, "pipeline": pipeline, "reward": reward})

    return records


# ── Dataset construction ───────────────────────────────────────────────────────

def build_dataset(
    records: list[dict],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build X (N, 26), y, sample_weights from replay records.

    Each record contains a single FeatureVector snapshot, so the std half of
    the 26-dim vector is set to zero during training. At inference time the
    controller receives a full window and computes real stds, giving the model
    additional signal as more data-driven models are trained.
    """
    X_rows, y_labels, raw_weights = [], [], []
    for rec in records:
        feat_dict = rec["features"]
        means = np.array([feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)
        stds = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        X_rows.append(np.concatenate([means, stds]))
        y_labels.append(rec["pipeline"])
        raw_weights.append(float(rec.get("reward", 1.0)))

    X = np.array(X_rows, dtype=np.float64)
    w = np.array(raw_weights, dtype=np.float64)
    w_shifted = w - w.min() + 1e-6
    w_normalized = w_shifted / w_shifted.sum()
    return X, y_labels, w_normalized


# ── Feature importance ─────────────────────────────────────────────────────────

def select_top_k_features(model, k: int) -> list[int]:
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature importances (top 26):")
    for rank, idx in enumerate(sorted_idx[:26], 1):
        bar = "#" * int(importances[idx] * 40)
        print(f"  {rank:2d}. {AGG_FEATURE_NAMES[idx]:35s} {importances[idx]:.4f}  {bar}")
    if k < 26:
        return sorted_idx[:k].tolist()
    return list(range(26))


# ── Model training ─────────────────────────────────────────────────────────────

def fit_and_save(
    X: np.ndarray,
    y: list[str],
    weights: np.ndarray,
    model_type: str,
    output_dir: str,
    top_k: int,
) -> None:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    print(f"\n{'=' * 60}")
    print(f"  Training: {model_type.upper()}")
    print(f"  Samples : {len(X)}  |  Features: {X.shape[1]}")
    print(f"{'=' * 60}")

    if model_type == "dt":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        filename = "dt_controller.joblib"
    else:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
        )
        filename = "rf_controller.joblib"

    try:
        X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
            X, y, weights, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

    model.fit(X_tr, y_tr, sample_weight=w_tr)
    print("\nClassification report (full 26 features):")
    print(classification_report(y_te, model.predict(X_te), zero_division=0))

    selected_features: list[int] | None = None
    if top_k < 26:
        top_k_idx = select_top_k_features(model, top_k)
        selected_features = top_k_idx
        model.fit(X_tr[:, selected_features], y_tr, sample_weight=w_tr)
        print(f"\nClassification report (top-{top_k} features):")
        print(classification_report(
            y_te, model.predict(X_te[:, selected_features]), zero_division=0
        ))
    else:
        select_top_k_features(model, 26)

    retained_names = (
        [AGG_FEATURE_NAMES[i] for i in selected_features]
        if selected_features is not None
        else AGG_FEATURE_NAMES
    )

    artifact = {
        "model": model,
        "selected_features": selected_features,
        "feature_names": retained_names,
        "pipeline_classes": list(model.classes_),
        "trained_on_n": len(X_tr),
    }

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"\nSaved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DT/RF meta-controllers from replay buffer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--replay",
        default="replay_buffer.jsonl",
        help="Path to replay buffer JSONL file",
    )
    p.add_argument(
        "--model",
        choices=["dt", "rf", "both"],
        default="both",
        help="Which model(s) to train",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "models"),
        help="Output directory for .joblib files",
    )
    p.add_argument(
        "--top-k-features",
        type=int,
        default=26,
        metavar="N",
        help="Keep only the top N most important features (1–26; 26 = keep all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records = load_replay_data(args.replay)
    if not records:
        print(f"No data found at '{args.replay}' — generating 500 synthetic samples.")
        records = generate_synthetic_data(500)
    else:
        print(f"Loaded {len(records)} records from '{args.replay}'.")

    X, y, weights = build_dataset(records)
    top_k = max(1, min(26, args.top_k_features))

    models_to_train = ["dt", "rf"] if args.model == "both" else [args.model]
    for mt in models_to_train:
        fit_and_save(X, y, weights, mt, args.output, top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
