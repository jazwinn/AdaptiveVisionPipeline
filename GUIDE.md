# Adaptive Vision Pipeline — User Guide

## Overview

The Adaptive Vision Pipeline is a real-time drone object detection system that automatically selects the best YOLO pipeline based on per-frame scene features (blur, motion, exposure, object sizes, etc.). It includes a PyQt5 GUI for easy configuration and live monitoring.

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the GUI

```bash
python Source/main.py
```

A window will open with a dark-themed control panel on the left and a video preview area on the right.

---

## GUI Layout & Controls

### Left Control Panel

#### **SOURCE** (top)
- **File button**: Browse for a single video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or image (`.jpg`, `.png`, etc.)
- **Dir button**: Browse for a folder containing images to process in sorted order
- **Path field**: Shows the selected source; you can also paste a path directly

#### **CONTROLLER**
- **Dropdown menu**: Choose the pipeline selection strategy
  - **rule**: Rule-based heuristics (fast baseline, CLAHE for low light, tiled for small objects, high-res for poor confidence)
  - **ucb**: Upper Confidence Bound multi-armed bandit (learns which pipeline works best)
  - **contextual**: Contextual bandit (learns per-feature-dimension weights)

#### **CONFIDENCE**
- **Slider**: YOLO detection confidence threshold (0.05 – 0.95)
- **Default**: 0.30 (30% confidence)
- Higher = fewer false positives, but may miss real objects
- Lower = more detections, but more false positives

#### **WINDOW SIZE (frames)**
- **Spinbox**: Number of frames over which to accumulate metrics before the controller decides which pipeline to use next
- **Default**: 30 frames
- Smaller windows = faster pipeline switches, more overhead
- Larger windows = smoother operation, but slower adaptation to scene changes

#### **MODE**
- **offline**: Process frames as fast as possible (no FPS limit)
- **realtime**: Process at a target FPS (useful for webcam or live streams)
- ⚠️ **Note**: Single images and image directories force **offline** mode

#### **OPTIONS** (checkboxes)
- **Save CSV log**: Save per-frame metrics (latency, confidence, pipeline choice, rewards) to `experiments/<timestamp>/results.csv`
- **Append replay buffer**: Store feature vectors and rewards for offline controller retraining
- **Fast-only pipeline**: Use only the fast baseline (PipelineA); skips heavy models
- **Include heavy (YOLOv8m)**: Add the highest-resolution YOLOv8m pipeline (costs ~4x latency, best quality)

#### **OUTPUT** (optional)
- **Path field**: Where to save annotated results
  - Single image input → save annotated image (e.g., `annotated.jpg`)
  - Directory input + `.mp4` extension → create a video montage
  - Directory input + folder path → save annotated images to that folder
- **Leave empty** to skip saving output; always displays results in the preview window

#### **START / STOP buttons**
- **START**: Validate source, load models, begin processing
- **STOP**: Gracefully halt processing mid-run

### Right Area: Video Preview
- Live annotated feed (boxes, track IDs, confidence scores)
- Text overlay: `[Pipeline] | [Latency] | [Reward]`
- Displays "Configure settings on the left, then press START" before processing

### Bottom Status Bar
- **Pipeline**: Current active pipeline name (`fast_baseline`, `clahe_pipeline`, `tiled_inference`, `high_res`)
- **Lat**: Per-frame inference latency (ms)
- **Avg**: Rolling 30-frame average latency
- **Dets**: Number of detections in the current frame
- **Reward**: Window-level reward score (+/−); higher is better
- **Frame**: Frame index in the video/sequence

---

## Typical Workflow

### Example 1: Analyze a video file

1. **Select source**: Click `File` → pick a `.mp4` or `.avi`
2. **Adjust settings**:
   - Controller: `rule` (fast, proven)
   - Confidence: `0.30` (default is fine)
   - Mode: `offline` (process as fast as possible)
3. **Optional output**: Set output path to save annotated video (e.g., `output.mp4`)
4. **Click START** → Watch the live feed and status bar update
5. **STOP** anytime to abort, or it will finish when done

### Example 2: Process a folder of drone frames

1. **Select source**: Click `Dir` → pick a folder with `.jpg` files
2. **Settings**:
   - Controller: `contextual` (learns from frames)
   - Window: `15` (adapt faster with fewer frames per decision)
   - Include heavy: ✓ (you want the best quality for static images)
3. **Output**: Pick a folder (e.g., `annotated_frames/`) to save each frame with detections
4. **Click START** → Processes all images in order
5. Status bar shows live metrics; window updates as frames complete

### Example 3: Find the best controller for your drone footage

1. **Run with controller=rule**; note the average reward in the status bar
2. **Run again with controller=ucb**; compare rewards
3. **Run with controller=contextual**; compare rewards
4. **Winner = your best controller for this footage**
5. Use `Log` checkbox to save detailed per-frame metrics (CSV) for analysis

---

## Understanding the Output

### Status Bar Metrics

| Field | Meaning | Good Range |
|-------|---------|------------|
| **Lat** | Single-frame inference time (ms) | 10–50 (depends on pipeline) |
| **Avg** | Smoothed latency over 30 frames | Stable trend is good |
| **Dets** | Objects found in this frame | Depends on scene |
| **Reward** | Quality (confidence) + Stability (track IDs) − Latency cost | +0.5 to +2.0 is healthy |
| **Frame** | Index in the source (0-based) | Increments per frame |

### Annotation Box Colors & Labels

- **Green boxes**: Object bounding boxes
- **Labels**: `#[track_id] [class_name] [confidence]`
  - `#1 person 0.92` = tracked person, 92% confidence, persistent ID #1
  - IDs persist across frames; new objects get new IDs

### Reward Signal

The reward combines three signals per window (default 30 frames):

- **Quality**: Mean detection confidence (higher = better)
- **Stability**: Penalizes flickering tracks (same object ID appearing/disappearing)
- **Cost**: Latency penalty (faster pipelines get a bonus)

**Formula**: `reward = quality + stability − latency_cost`, clipped to [−2, +3]

- Positive reward → good detection with low cost
- Negative reward → unreliable detections or very slow pipeline

---

## Controller Strategies Explained

### Rule-Based (Fastest, Most Interpretable)
- **If** image is very blurry → use CLAHE (contrast enhancement)
- **Else if** high optical flow (motion) → use fast baseline
- **Else if** many small objects (< 32×32 px) → use tiled inference
- **Else if** low confidence detections → use high-resolution
- **Else** → use fast baseline

✅ **Use when**: You want predictable, repeatable behavior  
❌ **Downside**: Cannot adapt to novel scene types

### UCB Bandit (Learns Best Single Pipeline)
- Treats each pipeline as an "arm" in a bandit problem
- Tracks mean reward per pipeline; explores arms with high uncertainty
- Over time, converges to the best single pipeline for your scenes

✅ **Use when**: One pipeline consistently outperforms  
❌ **Downside**: Slow to adapt if scene type changes (e.g., day → night)

### Contextual Bandit (Learns Per-Feature Weights)
- Uses 12 scene features (blur, flow, brightness, entropy, etc.) to weight pipelines
- Learns: "when blur is high AND flow is low, use CLAHE"
- More sophisticated than bandit; can adapt to multiple scene types

✅ **Use when**: Diverse scenes (mixed lighting, object sizes, motion)  
❌ **Downside**: Needs more frames to learn; requires logging for retraining

---

## Input Formats

### Video Files
Supported: `.mp4`, `.avi`, `.mov`, `.mkv`  
Automatically read at their native FPS.

### Single Image
Supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`  
Processed once; output is saved if output path is set.

### Image Directory
Supported: Folder with any mix of `.jpg`, `.png`, etc.  
Processed in alphabetical order; output can be:
- **Video montage** (if output ends in `.mp4`)
- **Image sequence** (if output is a folder path)

---

## Tips & Troubleshooting

### The GUI freezes while processing
- **This is normal**. Model loading takes 5–30 seconds. Watch the status bar; it will unfreeze once inference starts.

### Very high latency (100+ms)
- Your GPU may be overloaded or not in use
- Try `Fast-only` checkbox to disable heavy models
- Or reduce window size so it switches pipelines more often

### Detections are very noisy (many false positives)
- Increase confidence slider (e.g., 0.50 instead of 0.30)
- Or enable the `Include heavy` checkbox for the high-res pipeline

### Detections are too sparse (missing real objects)
- Decrease confidence slider
- Enable `Tiled inference` by ensuring `Fast-only` is unchecked
- Or enable `Include heavy` to use the best model

### Running out of VRAM
- Check `Fast-only` to use only small YOLOv8n
- Reduce window size (fewer frames in memory at once)

### Tracking IDs flicker (same object gets different IDs)
- This is a ByteTrack tuning issue, not a GUI issue
- The reward signal penalizes this; controllers will naturally avoid pipelines that cause flicker

### "No images found in directory"
- Check that the folder actually contains `.jpg`, `.png`, `.bmp`, `.tiff`, or `.webp` files
- No other files are supported

---

## Output Files

When you enable **Save CSV log**:

```
experiments/
└── <TIMESTAMP>/
    ├── config.json          # Run configuration (controller, window size, etc.)
    └── results.csv          # Per-frame metrics (frame, pipeline, latency, reward, etc.)
```

Open `results.csv` in Excel or Python to:
- Plot latency trends
- Compare pipeline usage
- Analyze reward over time
- Audit which frames were hard (low confidence)

---

## Example: Batch Processing Multiple Videos

```bash
# Process 3 videos with the same settings, save results
for video in footage_1.mp4 footage_2.mp4 footage_3.mp4; do
    python Source/main.py  # Configure GUI once, then run sequentially
done

# Results saved to experiments/<timestamp_1>/, experiments/<timestamp_2>/, etc.
```

Or use the GUI three times with different sources.

---

## Performance Tuning

| Goal | Setting |
|------|---------|
| **Maximum speed** | Controller: rule, Fast-only: ✓, Window: 20 |
| **Best quality** | Controller: contextual, Include heavy: ✓, Window: 60, Confidence: 0.50 |
| **Balanced** | Controller: rule, Fast-only: ☐, Window: 30, Confidence: 0.30 |
| **Learning mode** | Controller: contextual, Log: ✓, Window: 30+ (let it accumulate 100+ frames) |

---

## FAQ

**Q: Can I use the same model for real-time and offline mode?**  
A: Yes. Offline mode just processes as fast as possible; realtime mode paces output to match a target FPS.

**Q: What's the minimum viable input?**  
A: A single `.jpg` image. The GUI will process it once and save results if you set output path.

**Q: How do I compare controllers fairly?**  
A: Run the exact same video 3 times with different controllers, enable CSV logging, and compare the mean reward in `results.csv`.

**Q: Can I train the bandit controller offline?**  
A: The replay buffer collects feature vectors. A separate training script (not in this GUI) can use it. For now, the GUI only trains online.

**Q: Why does the status bar show different pipelines on successive frames?**  
A: The controller switches every N frames (window size). If you see `fast_baseline` → `clahe_pipeline` → `fast_baseline`, the scene features probably changed.

---

## Getting Help

Check the project README for architecture details.  
View the source code in `Source/` for implementation details.  
Error messages in the popup dialogs are usually clear; fix the input and try again.
