"""
GUI application for the Adaptive Vision Pipeline.
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

# torch / ultralytics must be imported before PyQt5 on Windows to avoid DLL
# initialization conflicts caused by Qt loading its own CRT first.
from ..core.config import RunConfig, RuntimeConfig
from ..core.frame_reader import FrameReader
from ..core.pipeline import Detection
from ..pipelines.pipeline_a import PipelineA
from ..pipelines.pipeline_b import PipelineB
from ..pipelines.pipeline_c import PipelineC
from ..pipelines.pipeline_d import PipelineD
from ..features.extractor import FeatureExtractor
from ..tracking.tracker import TrackerWrapper
from ..controller.rule_based import RuleBasedController
from ..controller.bandit import UCBBanditController, ContextualBanditController
from ..controller.decision_tree import DecisionTreeController, RandomForestController
from ..controller.orchestrator import PipelineOrchestrator
from ..evaluation.metrics import WindowMetrics, compute_reward
from ..evaluation.replay_buffer import ReplayBuffer
from ..experiments.logger import ExperimentLogger

import cv2
import numpy as np
import supervision as sv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout,
    QWidget, QFrame,
)

# ── Styles ────────────────────────────────────────────────────────────────────

_PANEL = "background-color: #2b2b2b;"
_INPUT = (
    "QLineEdit, QComboBox, QSpinBox {"
    "  background-color: #3c3c3c; color: #e8e8e8;"
    "  border: 1px solid #555; border-radius: 4px; padding: 4px 6px;"
    "}"
    "QComboBox::drop-down { border: none; }"
    "QComboBox QAbstractItemView { background: #3c3c3c; color: #e8e8e8; selection-background-color: #4a90d9; }"
)
_SLIDER = (
    "QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #4a90d9; width: 14px; height: 14px;"
    "  margin: -5px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #4a90d9; border-radius: 2px; }"
)
_BTN_START = (
    "QPushButton { background: #4caf50; color: #fff; border: none; border-radius: 6px;"
    "  padding: 9px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #43a047; }"
    "QPushButton:disabled { background: #444; color: #777; }"
)
_BTN_STOP = (
    "QPushButton { background: #e53935; color: #fff; border: none; border-radius: 6px;"
    "  padding: 9px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #c62828; }"
    "QPushButton:disabled { background: #444; color: #777; }"
)
_BTN_SMALL = (
    "QPushButton { background: #3c3c3c; color: #ccc; border: 1px solid #555;"
    "  border-radius: 4px; padding: 4px 8px; }"
    "QPushButton:hover { background: #4a4a4a; }"
)
_CHECK = "QCheckBox { color: #d0d0d0; spacing: 6px; }"
_RADIO = "QRadioButton { color: #d0d0d0; spacing: 6px; }"
_CAP = "color: #888; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
_STATUS = "color: #bbb; font-size: 12px; padding: 0 6px;"


def _cap(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_CAP)
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #444;")
    return line


# ── Background worker ─────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    frame_ready = pyqtSignal(object, object)   # (np.ndarray, dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self._run()
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    def _run(self):  # noqa: C901
        cfg = self.cfg
        source_path = cfg["source"]
        conf = cfg["conf"]

        pipelines = [PipelineA(conf=conf)]
        if not cfg["fast_only"]:
            pipelines.append(PipelineD(conf=conf))
            pipelines.append(PipelineC(conf=conf))
            if cfg["heavy"]:
                pipelines.append(PipelineB(conf=conf))
        pipeline_names = [p.name for p in pipelines]

        ctrl_name = cfg["controller"]
        if ctrl_name == "rule":
            controller = RuleBasedController()
        elif ctrl_name == "ucb":
            controller = UCBBanditController(pipeline_names)
        elif ctrl_name == "contextual":
            controller = ContextualBanditController(pipeline_names)
        elif ctrl_name == "decision_tree":
            controller = DecisionTreeController(pipeline_names)
        elif ctrl_name == "random_forest":
            controller = RandomForestController(pipeline_names)
        else:
            controller = ContextualBanditController(pipeline_names)

        mode = cfg["mode"]
        reader = FrameReader(
            source_path,
            max_fps=cfg["target_fps"] if mode == "realtime" else None,
        )
        if reader.source_type in ("image", "directory") and mode == "realtime":
            mode = "offline"

        runtime_cfg = RuntimeConfig(mode=mode, target_fps=cfg["target_fps"])
        orchestrator = PipelineOrchestrator(
            controller, pipelines,
            window_size=cfg["window"],
            runtime_config=runtime_cfg,
        )
        run_config = RunConfig(
            source_path=source_path,
            controller_type=ctrl_name,
            pipeline_names=pipeline_names,
            window_size=cfg["window"],
        )
        logger = ExperimentLogger(run_config) if cfg["log"] else None
        replay = ReplayBuffer() if cfg["replay"] else None
        extractor = FeatureExtractor()
        tracker = TrackerWrapper()
        window_metrics = WindowMetrics()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        latency_history: deque[float] = deque(maxlen=30)
        last_reward = 0.0

        out_writer = None
        out_dir = None
        out_image_path = None
        last_annotated = None
        output = cfg.get("output", "").strip()
        if output:
            if reader.source_type == "image":
                out_image_path = output
            elif reader.source_type == "directory":
                if Path(output).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_writer = cv2.VideoWriter(
                        output, fourcc, reader.fps, (reader.width, reader.height)
                    )
                else:
                    out_dir = Path(output)
                    out_dir.mkdir(parents=True, exist_ok=True)
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    output, fourcc, reader.fps, (reader.width, reader.height)
                )

        try:
            for frame in reader:
                if self._stop_event.is_set():
                    break

                features = extractor.extract(frame.image, [])
                dets, meta = orchestrator.process(frame, features)
                tracked = tracker.update(dets)
                window_metrics.update(tracked, meta["latency_ms"])
                latency_history.append(meta["latency_ms"])

                if (frame.index + 1) % cfg["window"] == 0:
                    episode = window_metrics.compute(orchestrator.current_pipeline_name)
                    last_reward = compute_reward(episode)
                    features_snap = (
                        orchestrator.feature_buffer[-1]
                        if orchestrator.feature_buffer else None
                    )
                    controller.update(
                        orchestrator.current_pipeline_name, last_reward, features_snap
                    )
                    if replay and features_snap:
                        replay.append(
                            features_snap,
                            orchestrator.current_pipeline_name,
                            last_reward,
                        )
                    window_metrics.reset()

                if logger:
                    logger.log_frame(
                        frame.index, meta["selected_pipeline"],
                        features, dets, last_reward, meta["latency_ms"],
                    )

                annotated = frame.image.copy()
                if tracked:
                    sv_dets = sv.Detections(
                        xyxy=np.array([t.bbox_xyxy for t in tracked]),
                        confidence=np.array([t.confidence for t in tracked]),
                        class_id=np.array([t.class_id for t in tracked]),
                    )
                    labels = [
                        f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                        for t in tracked
                    ]
                    annotated = box_annotator.annotate(annotated, sv_dets)
                    annotated = label_annotator.annotate(annotated, sv_dets, labels=labels)
                else:
                    sv_dets = sv.Detections.empty()
                cv2.putText(
                    annotated,
                    f"{meta['selected_pipeline']} | {meta['latency_ms']:.0f}ms"
                    f" | r={last_reward:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                last_annotated = annotated

                if out_writer:
                    out_writer.write(annotated)
                elif out_dir:
                    ext = Path(frame.source_path).suffix or ".jpg"
                    cv2.imwrite(str(out_dir / f"{frame.index:06d}{ext}"), annotated)

                stats = {
                    "pipeline": meta["selected_pipeline"],
                    "latency_ms": meta["latency_ms"],
                    "avg_latency_ms": float(np.mean(latency_history)),
                    "n_dets": len(dets),
                    "reward": last_reward,
                    "frame_idx": frame.index,
                }
                self.frame_ready.emit(annotated, stats)

        finally:
            reader.release()
            if out_writer:
                out_writer.release()
            if out_image_path and last_annotated is not None:
                cv2.imwrite(out_image_path, last_annotated)
            if logger:
                logger.save()


# ── Video display widget ──────────────────────────────────────────────────────

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__("Configure settings on the left, then press  START")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #111111; color: #555555; font-size: 14px;")

    def update_frame(self, img: np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Vision Pipeline")
        self.resize(1200, 760)
        self._worker: PipelineWorker | None = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_panel())
        root.addWidget(self._build_video_area())
        self._build_status_bar()

    # ── Left control panel ────────────────────────────────────────────────────

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(295)
        panel.setStyleSheet(_PANEL)
        v = QVBoxLayout(panel)
        v.setContentsMargins(14, 16, 14, 14)
        v.setSpacing(0)

        # ── Source ──
        v.addWidget(_cap("SOURCE"))
        v.addSpacing(4)
        src_row = QHBoxLayout()
        src_row.setSpacing(4)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("video / image / folder path")
        self.source_edit.setStyleSheet(_INPUT)
        src_row.addWidget(self.source_edit)
        btn_file = QPushButton("File")
        btn_file.setFixedWidth(44)
        btn_file.setStyleSheet(_BTN_SMALL)
        btn_file.clicked.connect(self._browse_source_file)
        btn_dir = QPushButton("Dir")
        btn_dir.setFixedWidth(38)
        btn_dir.setStyleSheet(_BTN_SMALL)
        btn_dir.clicked.connect(self._browse_source_dir)
        src_row.addWidget(btn_file)
        src_row.addWidget(btn_dir)
        v.addLayout(src_row)
        v.addSpacing(14)

        # ── Controller ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("CONTROLLER"))
        v.addSpacing(4)
        self.ctrl_combo = QComboBox()
        self.ctrl_combo.addItems(["rule", "ucb", "contextual", "decision_tree", "random_forest"])
        self.ctrl_combo.setStyleSheet(_INPUT)
        v.addWidget(self.ctrl_combo)
        v.addSpacing(14)

        # ── Confidence ──
        v.addWidget(_divider())
        v.addSpacing(10)
        self.conf_cap = _cap("CONFIDENCE  0.30")
        v.addWidget(self.conf_cap)
        v.addSpacing(4)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(30)
        self.conf_slider.setStyleSheet(_SLIDER)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_cap.setText(f"CONFIDENCE  {v / 100:.2f}")
        )
        v.addWidget(self.conf_slider)
        v.addSpacing(14)

        # ── Window size ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("WINDOW SIZE (frames)"))
        v.addSpacing(4)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 500)
        self.window_spin.setValue(30)
        self.window_spin.setStyleSheet(_INPUT)
        v.addWidget(self.window_spin)
        v.addSpacing(14)

        # ── Mode ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("MODE"))
        v.addSpacing(4)
        mode_row = QHBoxLayout()
        self.rb_offline = QRadioButton("offline")
        self.rb_realtime = QRadioButton("realtime")
        self.rb_offline.setChecked(True)
        for rb in (self.rb_offline, self.rb_realtime):
            rb.setStyleSheet(_RADIO)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        v.addLayout(mode_row)
        v.addSpacing(14)

        # ── Options ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("OPTIONS"))
        v.addSpacing(4)
        self.chk_log = QCheckBox("Save CSV log")
        self.chk_replay = QCheckBox("Append replay buffer")
        self.chk_fast_only = QCheckBox("Fast-only pipeline")
        self.chk_heavy = QCheckBox("Include heavy (YOLOv8m)")
        for chk in (self.chk_log, self.chk_replay, self.chk_fast_only, self.chk_heavy):
            chk.setStyleSheet(_CHECK)
            v.addWidget(chk)
        v.addSpacing(14)

        # ── Output ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("OUTPUT  (optional)"))
        v.addSpacing(4)
        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(".mp4 / folder / image path")
        self.output_edit.setStyleSheet(_INPUT)
        out_row.addWidget(self.output_edit)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(32)
        btn_out.setStyleSheet(_BTN_SMALL)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        v.addLayout(out_row)

        v.addStretch()

        # ── Start / Stop ──
        v.addWidget(_divider())
        v.addSpacing(10)
        self.start_btn = QPushButton("START")
        self.start_btn.setStyleSheet(_BTN_START)
        self.start_btn.clicked.connect(self._on_start)
        v.addWidget(self.start_btn)
        v.addSpacing(6)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet(_BTN_STOP)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        v.addWidget(self.stop_btn)

        return panel

    # ── Video area ────────────────────────────────────────────────────────────

    def _build_video_area(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #111111;")
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        self.video_label = VideoLabel()
        v.addWidget(self.video_label)
        return container

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #333;")
        self._sb_pipeline = QLabel("Pipeline: —")
        self._sb_latency  = QLabel("Lat: — ms")
        self._sb_avg      = QLabel("Avg: — ms")
        self._sb_dets     = QLabel("Dets: —")
        self._sb_reward   = QLabel("Reward: —")
        self._sb_frame    = QLabel("Frame: —")
        for w in (self._sb_pipeline, self._sb_latency, self._sb_avg,
                  self._sb_dets, self._sb_reward, self._sb_frame):
            w.setStyleSheet(_STATUS)
            sb.addWidget(w)

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_source_file(self):
        start = str(Path(self.source_edit.text()).parent) if self.source_edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source file", start,
            "Media (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All (*)",
        )
        if path:
            self.source_edit.setText(path)

    def _browse_source_dir(self):
        start = self.source_edit.text() if Path(self.source_edit.text()).is_dir() else ""
        path = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if path:
            self.source_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Output path", "",
            "Video (*.mp4);;All (*)",
        )
        if path:
            self.output_edit.setText(path)

    # ── Run control ───────────────────────────────────────────────────────────

    def _on_start(self):
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "No source", "Please select an input source.")
            return
        if not Path(source).exists():
            QMessageBox.warning(self, "Not found", f"Source not found:\n{source}")
            return

        cfg = {
            "source":     source,
            "controller": self.ctrl_combo.currentText(),
            "conf":       self.conf_slider.value() / 100.0,
            "window":     self.window_spin.value(),
            "mode":       "realtime" if self.rb_realtime.isChecked() else "offline",
            "target_fps": 15.0,
            "log":        self.chk_log.isChecked(),
            "replay":     self.chk_replay.isChecked(),
            "fast_only":  self.chk_fast_only.isChecked(),
            "heavy":      self.chk_heavy.isChecked(),
            "output":     self.output_edit.text().strip(),
        }

        self._worker = PipelineWorker(cfg)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_label.setText("Loading model…")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.stop_btn.setEnabled(False)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_frame(self, img, stats: dict):
        self.video_label.update_frame(img)
        self._sb_pipeline.setText(f"Pipeline: {stats['pipeline']}")
        self._sb_latency.setText(f"Lat: {stats['latency_ms']:.1f} ms")
        self._sb_avg.setText(f"Avg: {stats['avg_latency_ms']:.1f} ms")
        self._sb_dets.setText(f"Dets: {stats['n_dets']}")
        self._sb_reward.setText(f"Reward: {stats['reward']:+.3f}")
        self._sb_frame.setText(f"Frame: {stats['frame_idx']}")

    def _on_done(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Run complete.", 4000)

    def _on_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Pipeline error", msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(40, 40, 40))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(74, 144, 217))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
