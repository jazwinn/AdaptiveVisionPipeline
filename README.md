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

A window opens with a dark-themed control panel on the left and a video preview area on the right.

1. Enter a video file or folder path in **SOURCE**
2. Choose a controller from the **CONTROLLER** dropdown
3. Click **START**

---

## GUI Layout & Controls

### Left Control Panel

#### SOURCE
- **File button** — Browse for a single video (`.mp4`, `.avi`, `.mov`, `.mkv`) or image (`.jpg`, `.png`, etc.)
- **Dir button** — Browse for a folder of images processed in sorted order
- **Path field** — Shows the selected source; you can also paste a path directly

#### CONTROLLER
Dropdown to choose the pipeline selection strategy. See [Meta-Controllers](#meta-controllers) for details.

#### CONFIDENCE
- Slider: YOLO detection confidence threshold (0.05–0.95), default 0.30
- Higher = fewer false positives; lower = more detections

#### WINDOW SIZE (frames)
- Number of frames accumulated before the controller makes its next pipeline decision
- Default: 30. Smaller = faster switching; larger = smoother, slower adaptation

#### MODE
- **offline** — Process frames as fast as possible (no FPS limit)
- **realtime** — Process at a target FPS (for webcam or live streams)
- Single images and image directories always force **offline** mode

#### OPTIONS
- **Save CSV log** — Save per-frame metrics to `experiments/<timestamp>/results.csv`
- **Append replay buffer** — Store feature vectors and rewards for offline controller training
- **Fast-only pipeline** — Use only the fast YOLOv8n baseline; skips all heavy models
- **Include heavy (YOLOv8m)** — Add the highest-resolution pipeline (~4× latency, best quality)

#### OUTPUT (optional)
- Single image input → save annotated image (e.g. `annotated.jpg`)
- Directory input + `.mp4` extension → create an annotated video montage
- Directory input + folder path → save annotated images to that folder
- Leave empty to skip saving; preview always displays in the window

#### START / STOP
- **START** — Validate source, load models, begin processing
- **STOP** — Gracefully halt mid-run

### Right Area: Video Preview (▶ Live tab)
- Live annotated feed with bounding boxes, track IDs, and confidence scores
- Text overlay: `[Pipeline] | [Latency ms] | [Reward]`

### Bottom Status Bar

| Field | Meaning | Good range |
|---|---|---|
| **Pipeline** | Current active pipeline name | — |
| **Lat** | Single-frame inference time (ms) | 10–50 |
| **Avg** | Rolling 30-frame average latency | Stable trend |
| **Dets** | Objects found this frame | Depends on scene |
| **Reward** | Window-level reward score | +0.5 to +2.0 healthy |
| **Frame** | Frame index in source (0-based) | Increments per frame |

---

## Typical Workflows

### Analyze a video file

1. Click **File** → pick a `.mp4` or `.avi`
2. Controller: `rule`, Confidence: `0.30`, Mode: `offline`
3. Optionally set an output path to save the annotated video
4. Click **START** → watch live feed and status bar
5. **STOP** anytime or let it finish

### Process a folder of drone frames

1. Click **Dir** → pick a folder with `.jpg` files
2. Controller: `contextual`, Window: `15`, Include heavy: ✓
3. Output: pick a folder (e.g. `annotated_frames/`) to save each frame
4. Click **START** → processes all images in alphabetical order

### Find the best controller for your footage

1. Run with each controller in turn; note the average reward in the status bar
2. Enable **Save CSV log** to record per-frame metrics for deeper comparison
3. Or use the **📊 Benchmark** tab to run all controllers side-by-side automatically

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

Seven controllers are available, ranging from handcrafted rules to deep reinforcement learning:

### Rule-Based
Deterministic heuristics. No training needed. Fast and fully interpretable.

```
mean_intensity < 60  AND  intensity_std < 25    →  clahe_pipeline
optical_flow_magnitude > 8.0                    →  fast_baseline
small_object_ratio > 0.5 OR edge_density > 0.15 →  tiled
mean_confidence < 0.35 AND detection_count < 3  →  high_res
default                                          →  fast_baseline
```

✅ Predictable, repeatable — ❌ Cannot adapt to novel scenes

### UCB Bandit
Tracks mean reward per pipeline. Explores underused pipelines via an Upper Confidence Bound, then exploits the highest-value choice. Converges to the single best pipeline over time.

✅ Good when one pipeline dominates — ❌ Slow if scene type varies

### Contextual Bandit
Like UCB, but value estimates are conditioned on 12 frame features. Uses linear regression to model `reward = f(features, pipeline)`. Adapts to mixed scene types.

✅ Diverse scenes — ❌ Needs more data to learn effectively

### Decision Tree
Offline-trained `sklearn.DecisionTreeClassifier` (max_depth=5). Falls back to rule-based logic when no `.joblib` model file exists.

### Random Forest
Offline-trained `sklearn.RandomForestClassifier` (n_estimators=50, max_depth=5). Ensemble of decision trees — more robust on noisy or small training sets. Same fallback behaviour.

### Neural Network
Offline-trained PyTorch MLP classifier (`26 → 128 → 64 → 4 pipelines`). Learns non-linear feature interactions beyond what decision trees can express. Falls back to rule-based if torch is not installed or no `.pt` model exists.

### Neural RL (DQN)
Deep Q-Network that learns a Q-function over (state, pipeline-action) pairs. Optimises for long-term cumulative reward rather than instant classification accuracy. Supports optional online fine-tuning during live inference via epsilon-greedy exploration. Same fallback behaviour as Neural Network.

---

## Training Learned Controllers

All four learned controllers auto-generate 500 synthetic training samples when no replay buffer exists, so first-run training works with zero real data.

### Decision Tree & Random Forest

```bash
python -m Source.controller.train_dt --model both
```

| Flag | Default | Description |
|---|---|---|
| `--replay` | `replay_buffer.jsonl` | Path to replay data |
| `--model` | `both` | `dt`, `rf`, or `both` |
| `--output` | `Source/controller/models/` | Where to save `.joblib` files |
| `--top-k-features` | `26` | Prune to top-K features by importance |

### Neural Network (MLP Classifier)

```bash
python -m Source.controller.train_nn
```

| Flag | Default | Description |
|---|---|---|
| `--replay` | `replay_buffer.jsonl` | Path to replay data |
| `--output` | `Source/controller/models/` | Where to save `nn_controller.pt` |
| `--epochs` | `100` | Training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch-size` | `32` | Mini-batch size |

### Neural RL (DQN)

```bash
python -m Source.controller.train_rl
```

Consecutive replay-buffer records are paired as `(s, a, r, s_next)` transitions for Bellman-target training.

| Flag | Default | Description |
|---|---|---|
| `--replay` | `replay_buffer.jsonl` | Path to replay data |
| `--output` | `Source/controller/models/` | Where to save `rl_controller.pt` |
| `--epochs` | `200` | Training epochs |
| `--gamma` | `0.95` | Discount factor (~20-window horizon) |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch-size` | `64` | Mini-batch size |

All trained models are saved to `Source/controller/models/` and loaded automatically at GUI startup.

---

## Benchmark Tab

The **📊 Benchmark** tab runs any combination of controllers on the same source and compares them side-by-side.

1. Enter your source path in the **SOURCE** field on the left panel
2. Switch to the **📊 Benchmark** tab
3. Check the controllers you want to test
4. Set window size and whether to include the heavy pipeline
5. Click **RUN BENCHMARK**

Results appear row by row as each controller finishes. Best mean reward is highlighted green; worst is red.

Click **SAVE CSV** to export the full per-window records:

```csv
controller_name,window_index,pipeline_name,reward,mean_latency_ms,mean_confidence,track_consistency,flicker_rate
rule,0,fast_baseline,1.12,44.3,0.71,0.93,0.02
neural_rl,0,tiled,1.31,47.2,0.74,0.95,0.01
...
```

---

## Replay Buffer

When **Append replay buffer** is enabled, every completed window is appended to `replay_buffer.jsonl`:

```json
{"timestamp": 1717027200.0, "features": {"mean_intensity": 112.4, "optical_flow_magnitude": 3.1, ...}, "pipeline": "fast_baseline", "reward": 1.35}
```

Accumulate this file over real inference sessions, then retrain any of the four learned controllers to get a model tailored to your specific video domain.

---

## Understanding the Output

### Annotation Overlays
- **Green bounding boxes** with labels: `#[track_id] [class] [confidence]`
- Example: `#1 person 0.92` = tracked person at 92% confidence, persistent ID #1
- IDs persist across frames; new objects receive new IDs

### Reward Signal Breakdown

| Component | Formula | Effect |
|---|---|---|
| Quality | `mean_confidence × 2.0` | Rewards confident detections |
| Stability | `track_consistency − flicker_rate × 3.0` | Penalises flickering track IDs |
| Cost | `−max(0, latency − 50ms) / 100` | Penalises slow pipelines |
| **Total** | clipped to `[−2.0, +3.0]` | |

- **+0.5 to +2.0** → healthy detection with low cost
- **Negative** → unreliable detections or very slow pipeline

### Output Files (with Save CSV log enabled)

```
experiments/
└── <TIMESTAMP>/
    ├── config.json     # Run configuration (controller, window size, etc.)
    └── results.csv     # Per-frame metrics (frame, pipeline, latency, reward, etc.)
```

Open `results.csv` in Excel or Python to plot latency trends, compare pipeline usage, and audit difficult frames.

---

## Input Formats

| Type | Supported formats | Notes |
|---|---|---|
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` | Processed at native FPS |
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` | Processed once |
| Directory | Any folder with supported images | Alphabetical order; output can be video or image sequence |

---

## Performance Tuning

| Goal | Settings |
|---|---|
| **Maximum speed** | Controller: `rule`, Fast-only: ✓, Window: 20 |
| **Best quality** | Controller: `contextual`, Include heavy: ✓, Window: 60, Confidence: 0.50 |
| **Balanced** | Controller: `rule`, Fast-only: ☐, Window: 30, Confidence: 0.30 |
| **Learning / data collection** | Controller: `contextual`, Append replay buffer: ✓, Window: 30+ |
| **RL fine-tuning** | Controller: `neural_rl` (with `online_finetune=True` in code), Window: 30 |

---

## Reward Signal

```python
reward = (mean_confidence × 2.0) + (track_consistency − flicker_rate × 3.0) − latency_penalty
```

Clipped to `[−2.0, 3.0]`. Computed by `WindowMetrics` over each N-frame window and fed back to the controller via `update()`.

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

For learned controllers (DT, RF, NN, RL), a 26-dim **aggregated vector** is used: `[mean_f₀…mean_f₁₂, std_f₀…std_f₁₂]` computed over the current window.

---

## Project Layout

```
Source/
├── main.py
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
│   ├── neural_net.py             # MLP classifier controller
│   ├── neural_rl.py              # DQN reinforcement learning controller
│   ├── orchestrator.py           # PipelineOrchestrator (main loop glue)
│   ├── train_dt.py               # Train DT / RF from replay buffer
│   ├── train_nn.py               # Train Neural Network from replay buffer
│   ├── train_rl.py               # Train DQN from replay buffer
│   └── models/                   # Trained artifacts (.joblib, .pt)
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

Register it in `Source/gui/app.py`: add to the dropdown `addItems(...)` and the `if/elif` instantiation block. Also add it to `build_controllers()` in `Source/evaluation/benchmark.py`.

### Add a new pipeline

```python
# Source/pipelines/pipeline_e.py
from ..core.pipeline import DetectionPipeline, Detection
from ..core.frame_reader import Frame

class PipelineE(DetectionPipeline):
    name = "my_pipeline"
    cost_estimate = 30.0  # ms estimate for real-time budget clamping

    def run(self, frame: Frame) -> tuple[list[Detection], dict]:
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
| `torch` | Neural network + DQN training and inference |
| `PyQt5` | Desktop GUI |

`torch` is optional at runtime — `NeuralNetController` and `NeuralRLController` fall back to rule-based logic if it is not installed.

---

## Tips & Troubleshooting

**The GUI freezes while processing**
Normal — model loading takes 5–30 seconds. Watch the status bar; it will update once inference starts.

**Very high latency (100+ ms)**
- Try the **Fast-only** checkbox to disable heavy models
- Reduce window size so pipeline switches happen more frequently

**Too many false positives**
- Increase the confidence slider (e.g. 0.50)
- Enable **Include heavy** for the high-res pipeline

**Missing real objects (sparse detections)**
- Decrease confidence slider
- Uncheck **Fast-only** to enable tiled inference
- Enable **Include heavy**

**Tracking IDs flicker (same object gets different IDs)**
- ByteTrack tuning issue; the reward signal penalises this, so learned controllers naturally avoid pipelines that cause it

**"No images found in directory"**
- Check the folder contains `.jpg`, `.png`, `.bmp`, `.tiff`, or `.webp` files

**Neural net / RL controller does nothing different from rule-based**
- The `.pt` model file is missing — run `python -m Source.controller.train_nn` or `train_rl` first
- Or accumulate real replay buffer data for a more domain-specific model

---

## Notes

- **No GPU required.** Everything runs on CPU. YOLOv8 uses CUDA/MPS automatically if available.
- **All learned models are optional.** The GUI works at all times via rule-based fallback.
- **Synthetic training data is built in.** All training scripts generate 500 labelled samples automatically if no replay buffer exists.
- **The orchestrator enforces a latency budget** in `realtime` mode. If the controller picks a pipeline that exceeds `RuntimeConfig.max_pipeline_cost`, the cheapest affordable alternative is used instead.

---

## FAQ

**Q: Do I need a GPU?**
No. All components run on CPU. GPU inference is used automatically if available.

**Q: Which controller should I start with?**
`rule` for immediate use. Run the **Benchmark** tab to compare all options on your footage.

**Q: How do I train the neural controllers on my own data?**
Run the pipeline with **Append replay buffer** enabled to accumulate `replay_buffer.jsonl`, then run the relevant training script (`train_dt.py`, `train_nn.py`, or `train_rl.py`).

**Q: Can I use offline and realtime modes with the same model?**
Yes. `offline` processes as fast as possible; `realtime` paces output to a target FPS. The model and controller are identical.

**Q: What's the minimum viable input?**
A single `.jpg` image. The system processes it once and saves results if an output path is set.

**Q: How do I compare controllers fairly?**
Use the **📊 Benchmark** tab — it runs all selected controllers on the same source with the same settings and produces a side-by-side reward and latency table.
