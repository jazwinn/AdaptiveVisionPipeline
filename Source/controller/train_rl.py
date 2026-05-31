"""
Train the DQN Reinforcement Learning meta-controller from replay buffer data.

Usage:
    python -m Source.controller.train_rl [options]
    python train_rl.py [options]                # run from Source/controller/

Consecutive replay-buffer records are paired as (s, a, r, s_next) transitions.
If no replay buffer file is found, 500 synthetic samples are generated and
paired automatically so training works out-of-the-box without real data.

Outputs:
    rl_controller.pt   -- Saved to --output directory (default: models/)
"""
from __future__ import annotations

import argparse
import json
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

_HIDDEN_DIMS = [128, 128, 64]


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


# ── Transition construction ────────────────────────────────────────────────────

def _record_to_state(record: dict, rng: np.random.Generator) -> np.ndarray:
    """Convert one replay record → 26-dim state vector (stds = small noise)."""
    feat_dict = record["features"]
    means = np.array(
        [feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64
    )
    stds = np.abs(rng.normal(0.0, 0.01, size=len(FEATURE_NAMES)))
    return np.concatenate([means, stds]).astype(np.float32)


def build_transitions(
    records: list[dict],
    pipeline_names: list[str],
) -> list[tuple[np.ndarray, int, float, np.ndarray]]:
    """
    Build (s, a, r, s_next) transition tuples from sequential JSONL records.

    Consecutive records form time-ordered (state, next_state) pairs.
    The last record is discarded (no successor).
    """
    label2idx = {name: i for i, name in enumerate(pipeline_names)}
    rng = np.random.default_rng(0)
    transitions: list[tuple[np.ndarray, int, float, np.ndarray]] = []

    for i in range(len(records) - 1):
        rec      = records[i]
        rec_next = records[i + 1]

        pipeline = rec.get("pipeline", "")
        if pipeline not in label2idx:
            continue  # skip unknown pipeline labels

        s      = _record_to_state(rec, rng)
        a      = label2idx[pipeline]
        r      = float(rec.get("reward", 0.0))
        s_next = _record_to_state(rec_next, rng)
        transitions.append((s, a, r, s_next))

    return transitions


# ── DQN training ───────────────────────────────────────────────────────────────

def train_dqn(
    transitions: list[tuple[np.ndarray, int, float, np.ndarray]],
    pipeline_names: list[str],
    epochs: int = 200,
    batch_size: int = 64,
    gamma: float = 0.95,
    lr: float = 1e-3,
    target_update_freq: int = 100,
    val_fraction: float = 0.2,
    patience: int = 20,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    import torch
    import torch.nn as nn

    state_dim  = 26
    action_dim = len(pipeline_names)

    # Unpack transitions
    states, actions, rewards, next_states = zip(*transitions)
    S  = np.array(states,      dtype=np.float32)
    A  = np.array(actions,     dtype=np.int64)
    R  = np.array(rewards,     dtype=np.float32)
    NS = np.array(next_states, dtype=np.float32)

    # Train / val split
    n_val = max(1, int(len(S) * val_fraction))
    perm  = np.random.default_rng(42).permutation(len(S))
    vi, ti = perm[:n_val], perm[n_val:]

    S_tr, A_tr, R_tr, NS_tr = S[ti], A[ti], R[ti], NS[ti]
    S_val, A_val, R_val, NS_val = S[vi], A[vi], R[vi], NS[vi]

    def _build_net() -> nn.Module:
        layers: list[nn.Module] = []
        prev = state_dim
        for h in _HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        return nn.Sequential(*layers)

    q_online = _build_net()
    q_target = _build_net()
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = torch.optim.Adam(q_online.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()   # Huber loss

    rng_np = np.random.default_rng(99)
    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0
    total_steps = 0

    n_tr = len(S_tr)
    n_batches_per_epoch = max(1, n_tr // batch_size)

    print(f"\n{'=' * 60}")
    print(f"  Training: DQN Reinforcement Learning Controller")
    print(f"  Transitions : {len(S_tr)} train  |  {len(S_val)} val")
    print(f"  Actions     : {pipeline_names}")
    print(f"  Arch        : {state_dim} → {' → '.join(str(h) for h in _HIDDEN_DIMS)} → {action_dim}")
    print(f"  gamma={gamma}  lr={lr}  batch={batch_size}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        q_online.train()
        epoch_loss = 0.0

        # Shuffle training data each epoch
        ep_perm = rng_np.permutation(n_tr)

        for b in range(n_batches_per_epoch):
            idx = ep_perm[b * batch_size : (b + 1) * batch_size]
            if len(idx) == 0:
                continue

            s_b  = torch.from_numpy(S_tr[idx])
            a_b  = torch.from_numpy(A_tr[idx])
            r_b  = torch.from_numpy(R_tr[idx])
            ns_b = torch.from_numpy(NS_tr[idx])

            # Current Q-values for chosen actions
            q_pred = q_online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

            # Bellman target
            with torch.no_grad():
                q_next   = q_target(ns_b).max(dim=1).values
                q_target_val = r_b + gamma * q_next

            loss = criterion(q_pred, q_target_val)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_steps += 1

            if total_steps % target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())
                q_target.eval()

        # Validation loss
        q_online.eval()
        with torch.no_grad():
            s_vt  = torch.from_numpy(S_val)
            a_vt  = torch.from_numpy(A_val)
            r_vt  = torch.from_numpy(R_val)
            ns_vt = torch.from_numpy(NS_val)

            q_val_pred   = q_online(s_vt).gather(1, a_vt.unsqueeze(1)).squeeze(1)
            q_val_next   = q_target(ns_vt).max(dim=1).values
            q_val_target = r_vt + gamma * q_val_next
            val_loss     = criterion(q_val_pred, q_val_target).item()

        if epoch % 20 == 0 or epoch == 1:
            avg_train = epoch_loss / n_batches_per_epoch
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                f"steps={total_steps}"
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in q_online.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state:
        q_online.load_state_dict(best_state)

    # Sanity check: Q-value ranges per action on validation set
    q_online.eval()
    with torch.no_grad():
        q_all = q_online(torch.from_numpy(S_val)).numpy()

    print("\nQ-value ranges on validation set (sanity -check for overestimation):")
    for i, name in enumerate(pipeline_names):
        col = q_all[:, i]
        print(f"  {name:20s}  min={col.min():.3f}  mean={col.mean():.3f}  max={col.max():.3f}")

    print("\nGreedy policy distribution on validation set:")
    greedy_actions = q_all.argmax(axis=1)
    for i, name in enumerate(pipeline_names):
        frac = (greedy_actions == i).mean()
        bar  = "#" * int(frac * 30)
        print(f"  {name:20s}  {frac:.3f}  {bar}")

    return q_online


# ── Save artifact ──────────────────────────────────────────────────────────────

def save_model(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    pipeline_names: list[str],
    trained_on_n: int,
    gamma: float,
    output_dir: str,
) -> None:
    import torch

    out_path = Path(output_dir) / "rl_controller.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "q_state_dict":    model.state_dict(),
        "pipeline_names":  pipeline_names,
        "state_dim":       26,
        "action_dim":      len(pipeline_names),
        "hidden_dims":     _HIDDEN_DIMS,
        "feature_names":   AGG_FEATURE_NAMES,
        "epsilon":         0.15,        # starting epsilon for online fine-tuning
        "trained_on_n":    trained_on_n,
        "gamma":           gamma,
        "train_timestamp": time.time(),
    }
    torch.save(artifact, out_path)
    print(f"\nSaved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN RL meta-controller from replay buffer data.",
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
    p.add_argument("--epochs",     type=int,   default=200,  help="Training epochs")
    p.add_argument("--batch-size", type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--gamma",      type=float, default=0.95, help="Discount factor")
    p.add_argument("--lr",         type=float, default=1e-3, help="Adam learning rate")
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

    transitions = build_transitions(records, PIPELINE_NAMES)
    if len(transitions) < 2:
        print("ERROR: Not enough transitions to train (need at least 2 records).")
        return

    print(f"Built {len(transitions)} (s, a, r, s') transitions.")

    model = train_dqn(
        transitions,
        pipeline_names=PIPELINE_NAMES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
    )

    save_model(
        model,
        pipeline_names=PIPELINE_NAMES,
        trained_on_n=len(transitions),
        gamma=args.gamma,
        output_dir=args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
