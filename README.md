# Adaptive Vision Pipeline

An intelligent object detection system that observes its own performance and switches between detection pipelines in real time. Instead of running a single fixed model, a **meta-controller** watches frame-level signals — motion, blur, lighting, confidence — and picks whichever pipeline will perform best for the current conditions.

---

## How It Works

Every **N frames** (default: 30), the system:

1. Extracts 13 perceptual features from the current frame window
2. Asks the active controller: *"given these conditions, which pipeline should we use?"*
3. Switches to that pipeline if it differs from the current one
4. Computes a **reward** score from the resulting detections
5. Feeds the reward back to the controller so it can learn

For **independent image directories**, each image is treated as its own window — the tracker and feature extractor reset between images, and a stability-free reward is used (no track-consistency or flicker-rate terms).

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

A window opens with a dark-themed control panel on the left and three tabs on the right.

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
Dropdown to choose the pipeline selection strategy. See [Meta-Controllers](#meta-controllers) for details. Select **none** to run `fast_baseline` on every frame without any routing logic.

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
- **Append replay buffer** — Store feature vectors and rewards in `replay_buffer.jsonl` for offline controller training
- **Fast-only pipeline** — Use only the fast YOLOv8n baseline; skips all preprocessing pipelines
- **Include heavy (YOLOv8m)** — Add the highest-resolution pipeline (~4× latency, best quality)

#### OUTPUT (optional)
- Single image input → save annotated image (e.g. `annotated.jpg`)
- Directory input + `.mp4` extension → create an annotated video montage
- Directory input + folder path → save annotated images to that folder
- Leave empty to skip saving; preview always displays in the window

#### START / STOP
- **START** — Validate source, load models, begin processing
- **STOP** — Gracefully halt mid-run

---

### Right Area Tabs

#### ▶ Live Tab — Video Preview

Live annotated feed with bounding boxes, track IDs, and confidence scores.  
Text overlay per frame: `[Pipeline] | [Latency ms] | [Reward]`

**Playback Controls** (appear below the video while running):

| Button | Action |
|---|---|
| **⏸ PAUSE** | Freeze the current frame; switches to ▶ RESUME |
| **▶ RESUME** | Continue playing from the current frame |
| **◀** | Step back one frame (available only while paused) |
| **▶** | Step forward one frame (available only while paused) |
| **Seek slider** | Drag to jump to any position in the video or image sequence |
| **Frame: X / Y** | Current frame index and total frame count |

> **Tip:** Pause on a difficult frame, step forward/back to find the exact frame where detection degrades, then note the pipeline and features shown in the status bar.

#### Bottom Status Bar

| Field | Meaning | Good range |
|---|---|---|
| **Pipeline** | Current active pipeline name | — |
| **Lat** | Single-frame inference time (ms) | 10–50 |
| **Avg** | Rolling 30-frame average latency | Stable trend |
| **Dets** | Objects found this frame | Depends on scene |
| **Reward** | Window-level reward score | +0.5 to +2.0 healthy |
| **Frame** | Frame index in source (0-based) | Increments per frame |

---

#### 📊 Benchmark Tab

Runs any combination of controllers on the **same source** and compares them side-by-side using the live reward signal (no ground-truth labels required).

1. Enter your source path in the SOURCE field on the left panel
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
```

---

#### 🎯 Validate Tab

Evaluates pipelines and controllers against a **labelled YOLOv8 dataset** (requires a `data.yaml` file and ground-truth `.txt` label files). Three modes selectable via radio buttons:

**Single Model**
Runs `YOLO.val()` directly on the selected model. Returns mAP50, mAP50-95, precision, recall, and a per-class breakdown. Use this to benchmark a model's raw accuracy before any pipeline processing.

**Pipeline Comparison**
Runs each selected pipeline end-to-end (preprocess → infer) on every image in the split and computes accuracy against ground-truth boxes. Results table:

| Column | Description |
|---|---|
| Name | Pipeline name |
| mAP50 | Mean Average Precision at IoU 0.50 |
| mAP50-95 | mAP averaged over IoU 0.50–0.95 |
| Precision | TP / (TP + FP) at selected confidence |
| Recall | TP / (TP + FN) at selected confidence |
| Latency (ms) | Mean per-image inference time |
| **Missed** | False negatives shown as `FN / GT` (e.g. `192 / 1024` = 192 labeled objects not detected) |

**Controller Comparison**
Lets each controller choose a pipeline per image (using single-frame feature extraction) and compares the resulting accuracy across all controllers. Includes the same columns as Pipeline Comparison plus a **Top Pipeline** column showing the most-chosen pipeline and its usage percentage.

> **Important — YAML path format:** Paths in `data_hard.yaml` for `_hard` splits must be relative to the YAML file itself with no leading `../`. Example:
> ```yaml
> val:  valid_hard/images    # correct
> test: test_hard/images     # correct
> val:  ../valid_hard/images # wrong — resolves outside the dataset folder
> ```
> If the image directory cannot be found, the GUI shows an error dialog with the exact resolved path rather than silently falling back to a different split.

**Validate Tab Settings**

| Setting | Description |
|---|---|
| Data YAML | Path to `data.yaml` or `data_hard.yaml` |
| Split | `val`, `valid`, `test`, or `train` |
| Confidence | Detection threshold applied when counting TP/FP/FN |
| Image size | Inference resolution (pixels) |

---

## Detection Pipelines

| Name | Model | Preprocessing | When it's best |
|---|---|---|---|
| `fast_baseline` | YOLOv8n | None | High-motion scenes; latency budget is tight |
| `high_res` | YOLOv8m | None | Low confidence; high-stakes detection |
| `tiled` | YOLOv8n | Tile + stitch | Dense scenes; small or distant objects |
| `clahe_pipeline` | YOLOv8n | CLAHE (contrast enhance) | Low-light, underexposed, low-contrast frames |
| `bright_pipeline` | YOLOv8n | Gamma correction (γ=0.55) + mild CLAHE | Overexposed / glare-heavy images |
| `denoise_pipeline` | YOLOv8n | Non-Local Means denoising (h=8) | Images with heavy sensor or Gaussian noise |

`bright_pipeline` compresses blown-out highlights before detection — sending an overexposed image to CLAHE would worsen it, not help.  
`denoise_pipeline` applies NLM denoising which smooths noise while preserving real edges, giving YOLO a cleaner signal on noisy images.

---

## Meta-Controllers

Seven controllers are available, ranging from handcrafted rules to deep reinforcement learning. All controllers fall back to rule-based logic when their model file is missing or PyTorch is not installed.

### none
No routing — `fast_baseline` is used for every frame. Useful as a baseline to compare against adaptive controllers.

### Rule-Based
Deterministic heuristics. No training needed. Fast and fully interpretable.

```
overexposed_ratio > 0.20  OR  mean_intensity > 180       →  bright_pipeline
intensity_std > 65                                        →  denoise_pipeline
intensity_std < 35  OR  underexposed_ratio > 0.15
  OR  laplacian_variance < 100                            →  clahe_pipeline
optical_flow_magnitude > 8.0                              →  fast_baseline
small_object_ratio > 0.5  OR  edge_density > 0.15        →  tiled
mean_confidence < 0.35  AND  detection_count < 3         →  high_res
default                                                   →  fast_baseline
```

Rules fire top-to-bottom; the first match wins. Overexposure is handled before CLAHE to prevent sending glare-heavy images to contrast enhancement.

✅ Predictable, repeatable — ❌ Cannot adapt to novel scenes

### UCB Bandit
Tracks mean reward per pipeline. Explores underused pipelines via an Upper Confidence Bound, then exploits the highest-value choice. Converges to the single best pipeline over time.

✅ Good when one pipeline dominates — ❌ Slow to adapt if scene type varies

### Contextual Bandit
Like UCB, but value estimates are conditioned on 13 frame features. Uses linear regression to model `reward = f(features, pipeline)`. Adapts to mixed scene types within a single run.

✅ Diverse scenes — ❌ Needs more observations to learn effectively

### Decision Tree
Offline-trained `sklearn.DecisionTreeClassifier` (max_depth=5). Falls back to rule-based logic when no `.joblib` model file exists.

### Random Forest
Offline-trained `sklearn.RandomForestClassifier` (n_estimators=50, max_depth=5). Ensemble of decision trees — more robust on noisy or small training sets. Same fallback behaviour.

### Neural Network
Offline-trained PyTorch MLP classifier (`26 → 128 → 64 → 6 pipelines`). Learns non-linear feature interactions beyond what decision trees can express. Falls back to rule-based if torch is not installed or no `.pt` model exists.

### Neural RL (DQN)
Deep Q-Network that learns a Q-function over (state, pipeline-action) pairs. Optimises for long-term cumulative reward rather than instant classification accuracy. Supports optional online fine-tuning during live inference via epsilon-greedy exploration. Same fallback behaviour as Neural Network.

---

## Training Learned Controllers

All four learned controllers auto-generate **500 synthetic training samples** when no replay buffer exists, so first-run training works with zero real data.

> **What "synthetic samples" means:** The training scripts generate random feature vectors (blur, brightness, noise metrics, etc.) and label each one using the same if/elif rules as `rule_based.py`. Controllers trained purely on synthetic data will behave similarly to the rule-based controller — they learn to replicate those rules, not to surpass them. To get controllers that genuinely outperform rule-based, collect a real replay buffer (see [Replay Buffer](#replay-buffer)) and retrain.

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

## Hard Dataset Tool

`harden_dataset.py` creates degraded copies of a YOLOv8 split so that differences between pipelines become visible during evaluation. Labels are copied unchanged (bounding boxes remain valid after image-level degradation). Originals are never modified.

```bash
python Source/evaluation/harden_dataset.py \
    --dataset-dir "Solar Panel.v4i.yolov8" \
    --split test valid \
    --difficulty hard
```

| Flag | Default | Description |
|---|---|---|
| `--dataset-dir` | *(required)* | Path to the YOLOv8 dataset folder containing `data.yaml` |
| `--split` | `test` | One or more splits to harden: `train`, `valid`, `test` |
| `--difficulty` | `hard` | `easy`, `medium`, or `hard` |
| `--seed` | `42` | Random seed for reproducibility |

**Degradation types applied (hard difficulty):**

| Transform | Effect |
|---|---|
| Brightness reduction (×0.35) | Simulate deep shadow / night |
| Gaussian blur (k=11, σ=3.5) | Motion blur or defocus |
| Gaussian noise (std=45) | Sensor noise |
| Haze overlay (α=0.55) | Fog or dust in air |
| Low contrast (α=0.45, β=70) | Flat, washed-out image |
| Overexposure (×2.4) | Glare from direct sunlight |

Each image receives 1–3 randomly selected degradations (deterministic per seed).

**Output structure:**

```
dataset-dir/
  data.yaml             (original — unchanged)
  data_hard.yaml        (written by the script)
  test/images/          (original)
  test_hard/images/     (degraded copies — same filenames)
  test_hard/labels/     (copy of test/labels/)
  valid_hard/images/
  valid_hard/labels/
```

Use `data_hard.yaml` in the **🎯 Validate** tab to compare how each pipeline or controller handles degraded conditions.

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
- Example: `#1 panel 0.87` = solar panel at 87% confidence, persistent track ID #1
- IDs persist across frames within a video; they reset between independent images

### Reward Signal

**Video / sequential frames:**

| Component | Formula | Effect |
|---|---|---|
| Quality | `mean_confidence × 2.0` | Rewards confident detections |
| Stability | `track_consistency − flicker_rate × 3.0` | Penalises flickering track IDs |
| Cost | `−max(0, latency − 50ms) / 100` | Penalises slow pipelines |
| **Total** | clipped to `[−2.0, +3.0]` | |

**Independent image directories** (stability terms omitted — track IDs have no meaning across unrelated images):

| Component | Formula |
|---|---|
| Quality | `mean_confidence × 2.0` |
| Cost | `−max(0, latency − 50ms) / 100` |
| **Total** | clipped to `[−1.0, +2.0]` |

- **+0.5 to +2.0** → healthy detection with low cost
- **Negative** → unreliable detections or very slow pipeline

### Output Files (with Save CSV log enabled)

```
experiments/
└── <TIMESTAMP>/
    ├── config.json     # Run configuration (controller, window size, etc.)
    └── results.csv     # Per-frame metrics (frame, pipeline, latency, reward, etc.)
```

---

## Typical Workflows

### Analyze a video file

1. Click **File** → pick a `.mp4` or `.avi`
2. Controller: `rule`, Confidence: `0.30`, Mode: `offline`
3. Optionally set an output path to save the annotated video
4. Click **START** → watch live feed and status bar
5. Use **⏸ PAUSE** to freeze on any frame; **◀ ▶** to step frame by frame

### Process a folder of drone frames

1. Click **Dir** → pick a folder with `.jpg` files
2. Controller: `contextual`, Window: `15`, Include heavy: checked
3. Output: pick a folder (e.g. `annotated_frames/`) to save each frame
4. Click **START** → processes all images in alphabetical order

### Evaluate against a labelled dataset

1. Open the **🎯 Validate** tab
2. Click the folder icon to select `data.yaml` (or `data_hard.yaml` for degraded conditions)
3. Choose `Pipeline Comparison` or `Controller Comparison`
4. Select the pipelines/controllers to test and click **RUN**
5. Review the **Missed** column (`FN / GT`) alongside mAP50 and Recall to understand how many labeled objects were not detected

### Generate and evaluate a hard dataset

```bash
# 1. Create degraded test/valid splits
python Source/evaluation/harden_dataset.py \
    --dataset-dir "Solar Panel.v4i.yolov8" \
    --split test valid --difficulty hard

# 2. Validate in the GUI using data_hard.yaml  (🎯 Validate tab)
# 3. Compare mAP50 / Missed counts between data.yaml and data_hard.yaml
```

### Collect real replay data and retrain controllers

1. Run the GUI on real footage with **Append replay buffer** enabled
2. After collecting enough sessions, retrain:
   ```bash
   python -m Source.controller.train_nn
   python -m Source.controller.train_rl
   python -m Source.controller.train_dt --model both
   ```
3. Restart the GUI — models load automatically

---

## Input Formats

| Type | Supported formats | Notes |
|---|---|---|
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` | Processed at native FPS; seek slider enabled |
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` | Processed once |
| Directory | Any folder with supported images | Alphabetical order; per-image extractor/tracker reset |

---

## Performance Tuning

| Goal | Settings |
|---|---|
| **Maximum speed** | Controller: `rule`, Fast-only: checked, Window: 20 |
| **Best quality** | Controller: `contextual`, Include heavy: checked, Window: 60, Confidence: 0.50 |
| **Balanced** | Controller: `rule`, Fast-only: unchecked, Window: 30, Confidence: 0.30 |
| **Learning / data collection** | Controller: `contextual`, Append replay buffer: checked, Window: 30+ |
| **RL fine-tuning** | Controller: `neural_rl` (online_finetune=True), Window: 30 |

---

## Feature Vector (13 dimensions)

Extracted per frame by `FeatureExtractor` using OpenCV:

| Feature | What it measures |
|---|---|
| `laplacian_variance` | Sharpness via Laplacian kernel variance |
| `fft_blur_score` | High-frequency energy in the FFT spectrum |
| `mean_intensity` | Average pixel brightness |
| `intensity_std` | Brightness variation — also a noise proxy at high values |
| `underexposed_ratio` | Fraction of pixels below brightness 30 |
| `overexposed_ratio` | Fraction of pixels above brightness 225 |
| `optical_flow_magnitude` | Dense Farneback optical flow mean magnitude |
| `frame_displacement` | Corner-patch motion vector magnitude |
| `mean_confidence` | Mean detection confidence in this frame |
| `detection_count` | Number of objects detected |
| `small_object_ratio` | Fraction of detections smaller than 32×32 px |
| `edge_density` | Canny edge pixel fraction |
| `entropy` | Shannon entropy of the pixel histogram |

For learned controllers (DT, RF, NN, RL), a **26-dim aggregated vector** is used: `[mean_f₀…mean_f₁₂, std_f₀…std_f₁₂]` computed over the current window.

---

## Project Layout

```
Source/
├── main.py
├── core/
│   ├── frame_reader.py           # Video / image / directory reader; read_at() for seeking
│   ├── config.py                 # RunConfig, RuntimeConfig dataclasses
│   └── pipeline.py               # DetectionPipeline ABC + Detection dataclass
├── pipelines/
│   ├── pipeline_a.py             # fast_baseline     (YOLOv8n, no preprocessing)
│   ├── pipeline_b.py             # high_res          (YOLOv8m, no preprocessing)
│   ├── pipeline_c.py             # tiled             (YOLOv8n + tile/stitch)
│   ├── pipeline_d.py             # clahe_pipeline    (CLAHE + YOLOv8n)
│   ├── pipeline_e.py             # bright_pipeline   (gamma correction + mild CLAHE + YOLOv8n)
│   └── pipeline_f.py             # denoise_pipeline  (NLM denoising + YOLOv8n)
├── features/
│   └── extractor.py              # FeatureVector (13 fields) + FeatureExtractor
├── controller/
│   ├── base.py                   # MetaController ABC
│   ├── none.py                   # Pass-through (always fast_baseline)
│   ├── rule_based.py             # Heuristic controller (6-pipeline routing)
│   ├── bandit.py                 # UCBBanditController + ContextualBanditController
│   ├── decision_tree.py          # DecisionTreeController + RandomForestController
│   ├── neural_net.py             # NeuralNetController (MLP, 6 output classes)
│   ├── neural_rl.py              # NeuralRLController (DQN, 6 actions)
│   ├── orchestrator.py           # PipelineOrchestrator (main loop glue)
│   ├── train_dt.py               # Train DT / RF from replay buffer
│   ├── train_nn.py               # Train Neural Network from replay buffer
│   ├── train_rl.py               # Train DQN from replay buffer
│   └── models/                   # Trained artifacts (.joblib, .pt)
├── tracking/
│   └── tracker.py                # TrackerWrapper (ByteTrack multi-object tracking)
├── evaluation/
│   ├── metrics.py                # WindowMetrics, EpisodeResult, compute_reward, compute_reward_image
│   ├── replay_buffer.py          # ReplayBuffer (JSONL append)
│   ├── benchmark.py              # Benchmark backend (build_pipelines, build_controllers, run_controller)
│   ├── pipeline_eval.py          # Per-pipeline and per-controller evaluation against labelled datasets
│   ├── validate.py               # YOLO.val() wrapper (Single Model mode)
│   ├── harden_dataset.py         # Create degraded dataset copies for stress-testing
│   └── ablation.py               # Offline ablation sweep
├── experiments/
│   └── logger.py                 # ExperimentLogger (CSV + config.json)
└── gui/
    └── app.py                    # PyQt5 main window: Live tab, Benchmark tab, Validate tab
```

---

## Extending the System

### Add a new pipeline

```python
# Source/pipelines/pipeline_g.py
from ..core.pipeline import DetectionPipeline, Detection
import cv2, numpy as np

class PipelineG(DetectionPipeline):
    name = "my_pipeline"
    cost_estimate = 2.0   # relative cost for real-time budget clamping

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image  # apply any preprocessing here

    def infer(self, image: np.ndarray) -> list[Detection]:
        ...  # run model, return list[Detection]
```

Then:
1. Import `PipelineG` in `Source/evaluation/benchmark.py` → add to `build_pipelines()`
2. Import in `Source/gui/app.py` → add to the pipeline worker construction block and `_DIST_PIPELINES`
3. Add routing logic to `Source/controller/rule_based.py` (and both `_rule_fallback` methods in `neural_net.py` / `neural_rl.py`)
4. Add `"my_pipeline"` to `PIPELINE_NAMES` in `train_dt.py`, `train_nn.py`, `train_rl.py`

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

Register it in `Source/gui/app.py` (dropdown + instantiation block) and in `build_controllers()` in `Source/evaluation/benchmark.py`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 inference |
| `opencv-python` | Feature extraction, optical flow, preprocessing |
| `numpy` | Numerical operations |
| `supervision` | Detection rendering, IoU matching for evaluation |
| `scikit-learn` | Decision tree / random forest training |
| `joblib` | Model serialisation (ships with sklearn) |
| `torch` | Neural network + DQN training and inference |
| `PyQt5` | Desktop GUI |
| `pyyaml` | Dataset YAML parsing |

`torch` is optional at runtime — `NeuralNetController` and `NeuralRLController` fall back to rule-based logic if it is not installed.

---

## Tips & Troubleshooting

**The GUI freezes while processing**
Normal — model loading takes 5–30 seconds. Watch the status bar; it will update once inference starts.

**All controllers show the same pipeline (e.g. fast_baseline on every image)**
Check that the `fast_baseline` probe detections are non-empty. If you are running on an image directory with very small objects, try lowering the confidence slider or enabling the tiled pipeline.

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

**Validate tab: "Image directory does not exist" error**
The path in your `data.yaml` or `data_hard.yaml` could not be resolved. Check:
- Paths for `_hard` splits should have no leading `../` (e.g. `test_hard/images`, not `../test_hard/images`)
- The directory actually exists on disk
- The error dialog shows the exact resolved path to help diagnose

**Validate tab: mAP50 looks identical between data.yaml and data_hard.yaml**
mAP50 integrates over all confidence thresholds — a model can still detect objects at very low confidence even on degraded images. Check the **Missed** column (`FN / GT`) and **Recall** instead: these reflect what the model misses at your chosen confidence threshold. Also verify the `test` key in `data_hard.yaml` points to `test_hard/images` (not the original `test/images`).

**Tracking IDs flicker (same object gets different IDs)**
ByteTrack behaviour on fast-moving or heavily occluded scenes. The reward signal penalises this, so learned controllers naturally prefer pipelines that produce stable detections.

**Neural net / RL controller behaves identically to rule-based**
Either: (a) the `.pt` model file is missing — run `python -m Source.controller.train_nn` or `train_rl` first; or (b) the model was trained only on synthetic data, which is labeled using rule-based heuristics and produces near-identical decisions. Collect real replay buffer data and retrain for genuine improvement.

**"No images found in directory"**
Check the folder contains `.jpg`, `.png`, `.bmp`, `.tiff`, or `.webp` files.

---

## Notes

- **No GPU required.** Everything runs on CPU. YOLOv8 uses CUDA/MPS automatically if available.
- **All learned models are optional.** The GUI works at all times via rule-based fallback.
- **Synthetic training data is built in.** All training scripts generate 500 labelled samples automatically if no replay buffer exists — but models trained this way replicate rule-based decisions, not improve on them.
- **Image directories reset state per image.** The tracker and feature extractor reset between each image so that optical flow and track IDs do not bleed across unrelated photos.
- **The orchestrator enforces a latency budget** in `realtime` mode. If the controller picks a pipeline that exceeds `RuntimeConfig.max_pipeline_cost`, the cheapest affordable alternative is used instead.

---

## FAQ

**Q: Do I need a GPU?**
No. All components run on CPU. GPU inference is used automatically if available.

**Q: Which controller should I start with?**
`rule` for immediate use with no setup. Run the **📊 Benchmark** tab to compare all options on your footage.

**Q: How do I train the neural controllers on my own data?**
Run the pipeline with **Append replay buffer** enabled to accumulate `replay_buffer.jsonl`, then run the relevant training script (`train_dt.py`, `train_nn.py`, or `train_rl.py`). The more real sessions you accumulate, the more domain-specific the trained model becomes.

**Q: What is the difference between the Benchmark tab and the Validate tab?**
The **📊 Benchmark** tab compares controllers on a live source using the reward signal — no labels needed. The **🎯 Validate** tab compares pipelines and controllers against ground-truth labels (a labelled YOLOv8 dataset), producing mAP, precision, recall, and missed-object counts.

**Q: Why does mAP50 not drop much on the hard dataset even though I can see missed objects?**
mAP50 averages precision-recall across all confidence thresholds, including very low ones (0.001). A model can still produce detections at conf=0.02 that count toward mAP even when the display threshold (default 0.25) hides them. Use the **Missed** column and **Recall** at your chosen confidence for a more practical measure.

**Q: Can I use offline and realtime modes with the same model?**
Yes. `offline` processes as fast as possible; `realtime` paces output to a target FPS. The model and controller are identical.

**Q: What's the minimum viable input?**
A single `.jpg` image. The system processes it once and saves results if an output path is set.

**Q: How do I compare controllers fairly?**
Use the **📊 Benchmark** tab for reward-based comparison on real footage, or the **🎯 Validate** tab Controller Comparison mode for mAP/recall/missed-object comparison against labelled data.
