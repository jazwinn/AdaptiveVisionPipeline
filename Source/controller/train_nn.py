"""
Train the Neural Network (MLP Classifier) meta-controller from replay buffer data.

Usage:
    python -m Source.controller.train_nn [options]
    python train_nn.py [options]               # run from Source/controller/

If no replay buffer file is found, 500 synthetic samples are generated
automatically using rule-based heuristics so training works out-of-the-box.

Outputs:
    nn_controller.pt   -- Saved to --output directory (default: models/)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

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

PIPELINE_NAMES: list[str] = ["fast_baseline", "clahe_pipeline", "tiled", "high_res"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_replay_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
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
    records: list[dict] = []
    for _ in range(n_samples):
        feat = {
            "laplacian_variance":     float(rng.uniform(10, 1500)),
            "fft_blur_score":         float(rng.uniform(0, 100)),
            "mean_intensity":         float(rng.uniform(20, 235)),
            "intensity_std":          float(rng.uniform(5, 80)),
            "underexposed_ratio":     float(rng.uniform(0, 0.5)),
            "overexposed_ratio":      float(rng.uniform(0, 0.3)),
            "optical_flow_magnitude": float(rng.uniform(0, 20)),
            "frame_displacement":     float(rng.uniform(0, 30)),
            "mean_confidence":        float(rng.uniform(0.1, 0.95)),
            "detection_count":        int(rng.integers(0, 30)),
            "small_object_ratio":     float(rng.uniform(0, 1)),
            "edge_density":           float(rng.uniform(0, 0.4)),
            "entropy":                float(rng.uniform(2, 8)),
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
    Build X (N, 26), y (N,), sample_weights (N,) from replay records.

    Std half of the 26-dim vector is set to zero during training (single
    snapshot per record).  Small Gaussian noise is added to the std half to
    prevent the network from hard-coding zeros as a signal.
    """
    rng = np.random.default_rng(0)
    X_rows, y_labels, raw_weights = [], [], []

    for rec in records:
        feat_dict = rec["features"]
        means = np.array(
            [feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64
        )
        # stds = 0 + small jitter to mitigate covariate shift at inference
        stds = rng.normal(0.0, 0.01, size=len(FEATURE_NAMES)).astype(np.float64)
        stds = np.abs(stds)  # keep non-negative
        X_rows.append(np.concatenate([means, stds]))
        y_labels.append(rec["pipeline"])
        raw_weights.append(float(rec.get("reward", 1.0)))

    X = np.array(X_rows, dtype=np.float64)
    w = np.array(raw_weights, dtype=np.float64)
    w_shifted = w - w.min() + 1e-6
    w_normalized = w_shifted / w_shifted.sum()
    return X, y_labels, w_normalized


# ── Training ───────────────────────────────────────────────────────────────────

def train_nn(
    X: np.ndarray,
    y: list[str],
    weights: np.ndarray,
    pipeline_names: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dims: list[int],
    val_fraction: float = 0.2,
    patience: int = 15,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    import torch
    import torch.nn as nn

    label2idx = {name: i for i, name in enumerate(pipeline_names)}
    n_classes = len(pipeline_names)
    n_samples = len(X)

    # Filter out records whose pipeline label is not in pipeline_names
    valid = [(i, label2idx[y[i]]) for i in range(n_samples) if y[i] in label2idx]
    if not valid:
        raise ValueError("No training records match the pipeline names.")
    indices, labels_int = zip(*valid)
    X_f = X[list(indices)].astype(np.float32)
    y_i = np.array(labels_int, dtype=np.int64)
    w_f = weights[list(indices)].astype(np.float32)

    # Train / val split
    n_val = max(1, int(len(X_f) * val_fraction))
    perm = np.random.default_rng(42).permutation(len(X_f))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr, y_tr, w_tr = X_f[tr_idx], y_i[tr_idx], w_f[tr_idx]
    X_val, y_val = X_f[val_idx], y_i[val_idx]

    X_tr_t  = torch.from_numpy(X_tr)
    y_tr_t  = torch.from_numpy(y_tr)
    w_tr_t  = torch.from_numpy(w_tr)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Build network
    layers: list[nn.Module] = []
    prev = X_tr.shape[1]
    dropouts = [0.3, 0.2] + [0.1] * max(0, len(hidden_dims) - 2)
    for h, d in zip(hidden_dims, dropouts):
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print(f"  Training: Neural Network MLP Classifier")
    print(f"  Samples : {len(X_tr)} train  |  {len(X_val)} val")
    print(f"  Classes : {pipeline_names}")
    print(f"  Arch    : {X_tr.shape[1]} → {' → '.join(str(h) for h in hidden_dims)} → {n_classes}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm_ep = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tr_t), batch_size):
            idx = perm_ep[start : start + batch_size]
            xb, yb, wb = X_tr_t[idx], y_tr_t[idx], w_tr_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = (criterion(logits, yb) * wb).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).mean().item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train_loss={epoch_loss/max(n_batches,1):.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Final accuracy report
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1).numpy()

    print("\nPer-class accuracy on validation set:")
    for cls_idx, cls_name in enumerate(pipeline_names):
        mask = y_val == cls_idx
        if mask.sum() == 0:
            print(f"  {cls_name:20s}  -(no validation samples)")
            continue
        acc = (preds[mask] == cls_idx).mean()
        print(f"  {cls_name:20s}  {acc:.3f}  ({mask.sum()} samples)")

    return model


# ── Save artifact ──────────────────────────────────────────────────────────────

def save_model(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    pipeline_names: list[str],
    hidden_dims: list[int],
    trained_on_n: int,
    output_dir: str,
) -> None:
    import torch

    out_path = Path(output_dir) / "nn_controller.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": model.state_dict(),
        "pipeline_classes":  pipeline_names,
        "input_dim":         26,
        "hidden_dims":       hidden_dims,
        "feature_names":     AGG_FEATURE_NAMES,
        "trained_on_n":      trained_on_n,
        "train_timestamp":   time.time(),
    }
    torch.save(artifact, out_path)
    print(f"\nSaved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Neural Network MLP meta-controller from replay buffer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--replay",
        default="replay_buffer.jsonl",
        help="Path to replay buffer JSONL file",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "models"),
        help="Output directory for .pt file",
    )
    p.add_argument("--epochs",     type=int,   default=100,  help="Training epochs")
    p.add_argument("--lr",         type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--batch-size", type=int,   default=32,   help="Mini-batch size")
    return p.parse_args()


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        return

    args = parse_args()

    records = load_replay_data(args.replay)
    if not records:
        print(f"No data found at '{args.replay}' -generating 500 synthetic samples.")
        records = generate_synthetic_data(500)
    else:
        print(f"Loaded {len(records)} records from '{args.replay}'.")

    X, y, weights = build_dataset(records)
    hidden_dims = [128, 64]

    model = train_nn(
        X, y, weights,
        pipeline_names=PIPELINE_NAMES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=hidden_dims,
    )

    save_model(
        model,
        pipeline_names=PIPELINE_NAMES,
        hidden_dims=hidden_dims,
        trained_on_n=len(X),
        output_dir=args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
