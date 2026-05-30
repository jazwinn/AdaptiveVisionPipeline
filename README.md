# Adaptive Vision Pipeline

An intelligent object detection system that observes its own performance and switches between detection pipelines in real time. Instead of running a single fixed model, a **meta-controller** watches frame-level signals — motion, blur, lighting, confidence — and picks whichever pipeline will perform best for the current conditions.

---

## How It Works

Every **N frames** (default: 30), the system:

1. Extracts 13 perceptual features from the current frame window
2. Asks the active controller: *"given these conditions, which pipeline should we use?"*
3. Switches to that pipeline if it's different from the current one
4. Computes a **reward** score from the resulting detections
5. Feeds the reward back to the controller so it can learn

This loop runs in real time during live inference and can also be replayed offline for training and benchmarking.

---

## Detection Pipelines

| Name | Model | When it's best |
|---|---|---|
| `fast_baseline` | YOLOv8n | High-motion scenes; latency matters |
| `high_res` | YOLOv8m | Low confidence; high-stakes detection |
| `tiled` | YOLOv8n + tiling | Dense scenes; small object detection |
| `clahe_pipeline` | CLAHE + YOLOv8n | Low-light; low-contrast frames |

---

## Meta-Controllers

Five controllers are available, ranging from handcrafted rules to learned models:

### Rule-Based
Deterministic heuristics. No training needed.

```
mean_intensity < 60  AND  intensity_std < 25   →  clahe_pipeline
optical_flow_magnitude > 8.0                   →  fast_baseline
small_object_ratio > 0.5 OR edge_density > 0.15 →  tiled
mean_confidence < 0.35 AND detection_count < 3  →  high_res
default                                         →  fast_baseline
```

### UCB Bandit
Maintains a value estimate for each pipeline. Explores underused pipelines via an Upper Confidence Bound, then exploits the highest-value choice.

### Contextual Bandit
Like UCB, but value estimates are conditioned on frame features. Uses linear regression to model `reward = f(features, pipeline)`.

### Decision Tree
Offline-trained `sklearn.DecisionTreeClassifier` (max_depth=5). Trained on the replay buffer (or synthetic data if none exists). Falls back to rule-based logic when no `.joblib` model file is found.

### Random Forest
Offline-trained `sklearn.RandomForestClassifier` (n_estimators=50, max_depth=5). Ensemble of decision trees — more robust on noisy or small training sets. Same fallback behaviour as Decision Tree.

---

## Reward Signal

```python
reward = (mean_confidence × 2.0) + (track_consistency − flicker_rate × 3.0) − latency_penalty
```

Clipped to `[−2.0, 3.0]`. The reward penalises:
- Low detection confidence
- Track identity flicker (objects disappearing and reappearing between frames)
- Inference time above 50 ms

---

## Feature Vector (13 dimensions)

Extracted per frame by `FeatureExtractor` using OpenCV:

| Feature | What it measures |
|---|---|
| `laplacian_variance` | Sharpness via Laplacian kernel variance |
| `fft_blur_score` | High-frequency energy in the FFT spectrum |
| `mean_intensity` | Average pixel brightness |
| `intensity_std` | Brightness variation (contrast proxy) |
| `underexposed_ratio` | Fraction of pixels < 30 brightness |
| `overexposed_ratio` | Fraction of pixels > 225 brightness |
| `optical_flow_magnitude` | Dense Farneback optical flow mean magnitude |
| `frame_displacement` | Corner-patch motion vector magnitude |
| `mean_confidence` | Mean detection confidence in this frame |
| `detection_count` | Number of objects detected |
| `small_object_ratio` | Fraction of detections smaller than 32×32 px |
| `edge_density` | Canny edge pixel fraction |
| `entropy` | Shannon entropy of the pixel histogram |

For tree-based controllers, a 26-dim **aggregated vector** is used at decision time: `[mean_f₀, …, mean_f₁₂, std_f₀, …, std_f₁₂]` computed over the current window.

---

## Getting Started

### Install

```bash
git clone https://github.com/yourusername/AdaptiveVisionPipeline.git
cd AdaptiveVisionPipeline
pip install -r requirements.txt
```

### Launch the GUI

```bash
python -m Source.main
```

1. Enter a video file or folder path in **SOURCE**
2. Choose a controller from the **CONTROLLER** dropdown
3. Click **START**

The **▶ Live** tab shows the real-time video feed with detection overlays and a live metrics readout.

### Train Decision Tree / Random Forest

```bash
# With replay buffer data
python -m Source.controller.train_dt --replay replay_buffer.jsonl --model both

# Without any data (generates 500 synthetic samples automatically)
python -m Source.controller.train_dt --model both
```

Models are saved to `Source/controller/models/` and loaded automatically at GUI startup.

**Training options:**

| Flag | Default | Description |
|---|---|---|
| `--replay` | `replay_buffer.jsonl` | Path to accumulated replay data |
| `--model` | `both` | `dt`, `rf`, or `both` |
| `--output` | `Source/controller/models/` | Where to save `.joblib` files |
| `--top-k-features` | `26` | Prune to top-K features by importance |

---

## Benchmark Tab

The **📊 Benchmark** tab runs any combination of controllers on the same source and compares them side-by-side.

**To use it:**
1. Enter your source path in the main **SOURCE** field
2. Switch to the **📊 Benchmark** tab
3. Check the controllers you want to test
4. Set window size and whether to include the heavy pipeline (YOLOv8m)
5. Click **RUN BENCHMARK**

Results appear in a table as each controller finishes. The best mean reward is highlighted green; the worst is red.

Click **SAVE CSV** to export the full per-window records:

```csv
controller_name,window_index,pipeline_name,reward,mean_latency_ms,mean_confidence,track_consistency,flicker_rate
rule,0,fast_baseline,1.12,44.3,0.71,0.93,0.02
contextual,0,tiled,1.28,48.1,0.74,0.94,0.01
...
```

---

## Replay Buffer

When "Record replay buffer" is enabled, every completed window is appended to `replay_buffer.jsonl`:

```json
{"timestamp": 1717027200.0, "features": {"mean_intensity": 112.4, "optical_flow_magnitude": 3.1, ...}, "pipeline": "fast_baseline", "reward": 1.35}
```

This file is the training input for `train_dt.py`. Accumulate it over real inference sessions to build a data-driven tree controller tailored to your specific video domain.

---

## Project Layout

```
Source/
├── main.py                       # Entry point
├── core/
│   ├── frame_reader.py           # Video / image / directory reader
│   ├── config.py                 # RunConfig, RuntimeConfig dataclasses
│   └── pipeline.py               # DetectionPipeline ABC
├── pipelines/
│   ├── pipeline_a.py             # fast_baseline  (YOLOv8n)
│   ├── pipeline_b.py             # high_res       (YOLOv8m)
│   ├── pipeline_c.py             # tiled          (YOLOv8n + tiling)
│   └── pipeline_d.py             # clahe_pipeline (CLAHE + YOLOv8n)
├── features/
│   └── extractor.py              # FeatureVector + FeatureExtractor
├── controller/
│   ├── base.py                   # MetaController ABC
│   ├── rule_based.py             # Heuristic controller
│   ├── bandit.py                 # UCB + contextual bandit
│   ├── decision_tree.py          # DecisionTree + RandomForest controllers
│   ├── orchestrator.py           # PipelineOrchestrator (the main loop glue)
│   ├── train_dt.py               # Offline training script
│   └── models/                   # Trained .joblib artifacts (git-ignored)
├── tracking/
│   └── tracker.py                # Multi-object tracker (TrackerWrapper)
├── evaluation/
│   ├── metrics.py                # WindowMetrics, EpisodeResult, compute_reward
│   ├── replay_buffer.py          # ReplayBuffer (JSONL append)
│   ├── ablation.py               # Offline ablation sweep
│   └── benchmark.py              # Benchmark backend (GUI-agnostic)
├── experiments/
│   └── logger.py                 # ExperimentLogger
└── gui/
    └── app.py                    # PyQt5 main window + BenchmarkWidget
```

---

## Extending the System

### Add a new controller

```python
# Source/controller/my_controller.py
from .base import MetaController
from ..features.extractor import FeatureVector

class MyController(MetaController):
    window_size = 30

    def select_pipeline(self, feature_history: list[FeatureVector], pipeline_names: list[str]) -> str:
        # your logic here
        return pipeline_names[0]

    def update(self, pipeline_name: str, reward: float, features: FeatureVector | None = None) -> None:
        pass  # optional: online learning
```

Then register it in `Source/gui/app.py` — add it to the dropdown and the controller selection block.

### Add a new pipeline

```python
# Source/pipelines/pipeline_e.py
from ..core.pipeline import DetectionPipeline, Detection
from ..core.frame_reader import Frame

class PipelineE(DetectionPipeline):
    name = "my_pipeline"
    cost_estimate = 30.0  # ms estimate for budget clamping

    def run(self, frame: Frame) -> tuple[list[Detection], dict]:
        # your inference here
        return detections, {"latency_ms": ...}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 inference |
| `opencv-python` | Feature extraction, optical flow |
| `numpy` | Numerical operations |
| `supervision` | Detection rendering in GUI |
| `scikit-learn` | Decision tree / random forest training |
| `joblib` | Model serialisation (ships with sklearn) |
| `PyQt5` | Desktop GUI |

---

## Notes

- **No GPU required.** Everything runs on CPU. YOLOv8 will use CUDA/MPS automatically if available.
- **Tree models are optional.** The GUI works without `.joblib` files — both controllers fall back to rule-based logic.
- **Synthetic training data is always available.** `train_dt.py` generates 500 heuristic-labelled samples if no replay buffer file exists, so you can train and demo immediately.
- **The orchestrator enforces a cost budget** in `realtime` mode (`RuntimeConfig.max_pipeline_cost`). If the controller picks a pipeline that's too expensive, the cheapest affordable alternative is used instead.
