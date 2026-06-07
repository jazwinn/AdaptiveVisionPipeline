"""
GUI application for the Adaptive Vision Pipeline.
"""
from __future__ import annotations

import sys
import threading
import time
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
from ..pipelines.pipeline_e import PipelineE
from ..pipelines.pipeline_f import PipelineF
from ..features.extractor import FeatureExtractor
from ..tracking.tracker import TrackerWrapper
from ..controller.rule_based import RuleBasedController
from ..controller.bandit import UCBBanditController, ContextualBanditController
from ..controller.decision_tree import DecisionTreeController, RandomForestController
from ..controller.neural_net import NeuralNetController
from ..controller.neural_rl import NeuralRLController
from ..controller.none import NoneController
from ..controller.orchestrator import PipelineOrchestrator
from ..evaluation.metrics import WindowMetrics, compute_reward, compute_reward_image
from ..evaluation.replay_buffer import ReplayBuffer
from ..evaluation.benchmark import (
    BenchmarkResult, PerWindowRecord,
    build_pipelines, build_controllers, run_controller, write_csv,
)
from ..evaluation.validate import run_validation
from ..evaluation.pipeline_eval import run_pipeline_comparison, run_controller_comparison
from ..experiments.logger import ExperimentLogger
from .train_widget import TrainWidget

import cv2
import numpy as np
import supervision as sv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QButtonGroup, QProgressBar, QRadioButton, QSizePolicy, QSlider, QSpinBox,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
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
    finished    = pyqtSignal()
    error       = pyqtSignal(str)
    paused      = pyqtSignal(bool)             # True = paused, False = resumed
    frame_info  = pyqtSignal(int, int)         # (current_idx, total_frames)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg          = cfg
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        self._step_lock   = threading.Lock()
        self._step_delta  = 0
        self._seek_target = -1

    def stop(self):
        self._pause_event.clear()   # unblock the pause gate so the thread can exit
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()
        self.paused.emit(True)

    def resume(self):
        self._pause_event.clear()
        self.paused.emit(False)

    def step(self, delta: int):
        with self._step_lock:
            self._step_delta = delta

    def seek(self, idx: int):
        with self._step_lock:
            self._seek_target = idx

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

        model_path = cfg.get("model_path") or None
        # None triggers each pipeline's built-in default (models/ at project root)
        pipelines = [PipelineA(conf=conf, model_path=model_path)]
        if not cfg["fast_only"]:
            pipelines.append(PipelineD(conf=conf, model_path=model_path))
            pipelines.append(PipelineC(conf=conf, model_path=model_path))
            pipelines.append(PipelineE(conf=conf, model_path=model_path))
            pipelines.append(PipelineF(conf=conf, model_path=model_path))
            if cfg["heavy"]:
                pipelines.append(PipelineB(conf=conf, model_path=model_path))
        pipeline_names = [p.name for p in pipelines]

        ctrl_name = cfg["controller"]
        if ctrl_name == "none":
            controller = NoneController()
        elif ctrl_name == "rule":
            controller = RuleBasedController()
        elif ctrl_name == "ucb":
            controller = UCBBanditController(pipeline_names)
        elif ctrl_name == "contextual":
            controller = ContextualBanditController(pipeline_names)
        elif ctrl_name == "decision_tree":
            controller = DecisionTreeController(pipeline_names)
        elif ctrl_name == "random_forest":
            controller = RandomForestController(pipeline_names)
        elif ctrl_name == "neural_net":
            controller = NeuralNetController(pipeline_names)
        elif ctrl_name == "neural_rl":
            controller = NeuralRLController(pipeline_names)
        else:
            controller = ContextualBanditController(pipeline_names)

        mode = cfg["mode"]
        reader = FrameReader(
            source_path,
            max_fps=cfg["target_fps"] if mode == "realtime" else None,
        )
        if reader.source_type in ("image", "directory") and mode == "realtime":
            mode = "offline"
        is_image_mode = reader.source_type in ("image", "directory")

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
        if replay is not None:
            import sys
            print(f"[REPLAY] Buffer active → {replay.path}", flush=True, file=sys.stderr)
        extractor = FeatureExtractor()
        tracker = TrackerWrapper()
        window_metrics = WindowMetrics()
        box_annotator   = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator  = sv.MaskAnnotator(opacity=0.45)
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
            idx   = 0
            total = reader.frame_count

            while not self._stop_event.is_set() and idx < total:
                # ── Apply any pending seek (works during playback too) ────────
                with self._step_lock:
                    if self._seek_target >= 0:
                        idx = max(0, min(total - 1, self._seek_target))
                        self._seek_target = -1

                # ── Pause gate ───────────────────────────────────────────────
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    with self._step_lock:
                        if self._seek_target >= 0:
                            idx = max(0, min(total - 1, self._seek_target))
                            self._seek_target = -1
                            break
                        delta = self._step_delta
                        self._step_delta = 0
                    if delta != 0:
                        idx = max(0, min(total - 1, idx + delta))
                        break          # fall through to process this one frame
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    break

                frame = reader.read_at(idx)
                if frame is None:
                    idx += 1
                    continue

                features = extractor.extract(frame.image, [])
                dets, meta = orchestrator.process(frame, features)
                tracked = tracker.update(dets)
                window_metrics.update(tracked, meta["latency_ms"])
                latency_history.append(meta["latency_ms"])

                at_boundary = is_image_mode or ((frame.index + 1) % cfg["window"] == 0)
                if at_boundary:
                    episode = window_metrics.compute(orchestrator.current_pipeline_name)
                    last_reward = (
                        compute_reward_image(episode) if is_image_mode
                        else compute_reward(episode)
                    )
                    features_snap = (
                        orchestrator.feature_buffer[-1]
                        if orchestrator.feature_buffer else None
                    )
                    controller.update(
                        orchestrator.current_pipeline_name, last_reward, features_snap
                    )
                    if replay is not None and features_snap is not None:
                        replay.append(
                            features_snap,
                            orchestrator.current_pipeline_name,
                            last_reward,
                        )
                    window_metrics.reset()
                    if is_image_mode:
                        extractor.reset()
                        tracker.reset()

                if logger:
                    logger.log_frame(
                        frame.index, meta["selected_pipeline"],
                        features, dets, last_reward, meta["latency_ms"],
                    )

                annotated = frame.image.copy()
                if tracked:
                    has_masks = any(t.mask is not None for t in tracked)
                    if has_masks:
                        fh, fw = annotated.shape[:2]
                        mask_arr = np.zeros((len(tracked), fh, fw), dtype=bool)
                        for mi, t in enumerate(tracked):
                            if t.mask is not None:
                                mh, mw = t.mask.shape
                                mask_arr[mi, :mh, :mw] = t.mask
                    else:
                        mask_arr = None

                    sv_dets = sv.Detections(
                        xyxy=np.array([t.bbox_xyxy for t in tracked]),
                        confidence=np.array([t.confidence for t in tracked]),
                        class_id=np.array([t.class_id for t in tracked]),
                        mask=mask_arr,
                    )
                    labels = [
                        f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                        for t in tracked
                    ]
                    if has_masks:
                        annotated = mask_annotator.annotate(annotated, sv_dets)
                    else:
                        annotated = box_annotator.annotate(annotated, sv_dets)
                    annotated = label_annotator.annotate(annotated, sv_dets, labels=labels)
                else:
                    sv_dets = sv.Detections.empty()
                    has_masks = False

                seg_tag = " [SEG]" if has_masks else ""
                cv2.putText(
                    annotated,
                    f"{meta['selected_pipeline']}{seg_tag} | {meta['latency_ms']:.0f}ms"
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
                self.frame_info.emit(idx, total)
                self.frame_ready.emit(annotated, stats)

                # Only auto-advance when not paused; a stepped frame keeps idx
                if not self._pause_event.is_set():
                    idx += 1

        finally:
            reader.release()
            if out_writer:
                out_writer.release()
            if out_image_path and last_annotated is not None:
                cv2.imwrite(out_image_path, last_annotated)
            if logger:
                logger.save()


# ── Benchmark worker ─────────────────────────────────────────────────────────

class BenchmarkWorker(QThread):
    controller_done = pyqtSignal(object)          # BenchmarkResult
    progress        = pyqtSignal(str, int)        # (status_text, pct 0-100)
    finished_all    = pyqtSignal(object, object)  # (list[BenchmarkResult], list[PerWindowRecord])
    error           = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._run()
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _run(self):
        cfg = self.cfg
        pipelines = build_pipelines(
            conf=cfg["conf"],
            include_heavy=cfg["include_heavy"],
            model_path=cfg.get("model_path") or None,
        )
        pipeline_names = [p.name for p in pipelines]

        controllers, warns = build_controllers(cfg["selected"], pipeline_names)
        for w in warns:
            self.progress.emit(f"⚠ {w}", 0)

        all_results: list[BenchmarkResult] = []
        all_records: list[PerWindowRecord] = []
        n_total = len(controllers)

        for idx, (name, ctrl) in enumerate(controllers):
            if self._stop.is_set():
                break
            self.progress.emit(f"Running {name}… ({idx + 1}/{n_total})", int(idx / n_total * 100))

            def _cb(ctrl_name: str, frame_idx: int, _name=name):
                self.progress.emit(f"Running {_name}… frame {frame_idx}", int((_name and 0) or 0))

            result, records = run_controller(
                cfg["source"], ctrl, pipelines, cfg["window"],
                progress_cb=_cb,
            )
            # Overwrite controller_name with the short label used in the GUI
            result.controller_name = name
            for rec in records:
                rec.controller_name = name

            all_results.append(result)
            all_records.extend(records)
            self.controller_done.emit(result)

        self.progress.emit("Done", 100)
        self.finished_all.emit(all_results, all_records)


# ── Benchmark widget ──────────────────────────────────────────────────────────

_TABLE_COLS = [
    "Controller", "Mean Reward", "Std", "Min", "Max",
    "Latency (ms)", "P95 (ms)", "Switches", "Top Pipeline",
]

_DIST_PIPELINES = ["fast_baseline", "clahe_pipeline", "tiled", "high_res",
                   "bright_pipeline", "denoise_pipeline"]

_TBL_STYLE = (
    "QTableWidget {"
    "  background-color: #1e1e1e; color: #e0e0e0;"
    "  gridline-color: #3a3a3a; border: none;"
    "}"
    "QTableWidget::item { padding: 4px 8px; }"
    "QHeaderView::section {"
    "  background-color: #2b2b2b; color: #aaa;"
    "  border: none; padding: 4px 8px; font-weight: bold;"
    "}"
)


class BenchmarkWidget(QWidget):
    def __init__(self, source_getter, model_getter=None):
        """
        Parameters
        ----------
        source_getter : callable() → str
            Returns the source path from the main panel's SOURCE field.
        model_getter  : callable() → str, optional
            Returns the model path from the main panel's MODEL field.
            When None or when the returned string is empty, pipelines use
            their default weights.
        """
        super().__init__()
        self._source_getter = source_getter
        self._model_getter = model_getter or (lambda: "")
        self._worker: BenchmarkWorker | None = None
        self._all_records: list[PerWindowRecord] = []
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left control strip ────────────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(210)
        ctrl_panel.setStyleSheet(_PANEL)
        v = QVBoxLayout(ctrl_panel)
        v.setContentsMargins(12, 14, 12, 14)
        v.setSpacing(0)

        v.addWidget(_cap("CONTROLLERS"))
        v.addSpacing(4)
        self._chk_controllers: dict[str, QCheckBox] = {}
        for name in ("none", "rule", "ucb", "contextual", "decision_tree", "random_forest", "neural_net", "neural_rl"):
            chk = QCheckBox(name)
            chk.setChecked(name != "none")  # "none" unchecked by default
            chk.setStyleSheet(_CHECK)
            v.addWidget(chk)
            self._chk_controllers[name] = chk
        v.addSpacing(14)

        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("WINDOW SIZE (frames)"))
        v.addSpacing(4)
        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 500)
        self._window_spin.setValue(30)
        self._window_spin.setStyleSheet(_INPUT)
        v.addWidget(self._window_spin)
        v.addSpacing(10)

        self._chk_heavy = QCheckBox("Include heavy (YOLOv8m)")
        self._chk_heavy.setStyleSheet(_CHECK)
        v.addWidget(self._chk_heavy)
        v.addSpacing(14)

        v.addWidget(_divider())
        v.addSpacing(10)
        self._run_btn = QPushButton("RUN BENCHMARK")
        self._run_btn.setStyleSheet(_BTN_START)
        self._run_btn.clicked.connect(self._on_run)
        v.addWidget(self._run_btn)
        v.addSpacing(6)
        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setStyleSheet(_BTN_STOP)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        v.addWidget(self._stop_btn)
        v.addSpacing(6)
        self._save_btn = QPushButton("SAVE CSV")
        self._save_btn.setStyleSheet(_BTN_SMALL)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_csv)
        v.addWidget(self._save_btn)
        v.addSpacing(10)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #3c3c3c; border: none; border-radius: 3px; height: 8px; }"
            "QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }"
        )
        v.addWidget(self._progress_bar)
        v.addSpacing(4)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        v.addWidget(self._status_lbl)

        v.addStretch()
        root.addWidget(ctrl_panel)

        # ── Right: tables ─────────────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet("background-color: #1a1a1a;")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 10)
        rv.setSpacing(6)

        summary_lbl = QLabel("Summary")
        summary_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        rv.addWidget(summary_lbl)

        self._summary_table = QTableWidget(0, len(_TABLE_COLS))
        self._summary_table.setHorizontalHeaderLabels(_TABLE_COLS)
        self._summary_table.setStyleSheet(_TBL_STYLE)
        self._summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.verticalHeader().setVisible(False)
        rv.addWidget(self._summary_table, stretch=3)

        dist_lbl = QLabel("Pipeline Distribution (% of windows)")
        dist_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        rv.addWidget(dist_lbl)

        dist_cols = ["Controller"] + _DIST_PIPELINES
        self._dist_table = QTableWidget(0, len(dist_cols))
        self._dist_table.setHorizontalHeaderLabels(dist_cols)
        self._dist_table.setStyleSheet(_TBL_STYLE)
        self._dist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._dist_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._dist_table.horizontalHeader().setStretchLastSection(True)
        self._dist_table.verticalHeader().setVisible(False)
        rv.addWidget(self._dist_table, stretch=2)

        root.addWidget(right)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        source = self._source_getter()
        if not source:
            QMessageBox.warning(self, "No source", "Enter a source path in the SOURCE field first.")
            return

        selected = [name for name, chk in self._chk_controllers.items() if chk.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No controllers", "Select at least one controller.")
            return

        # Clear tables
        self._summary_table.setRowCount(0)
        self._dist_table.setRowCount(0)
        self._all_records = []
        self._save_btn.setEnabled(False)

        cfg = {
            "source":        source,
            "conf":          0.30,
            "window":        self._window_spin.value(),
            "include_heavy": self._chk_heavy.isChecked(),
            "selected":      selected,
            "model_path":    self._model_getter().strip(),
        }

        self._worker = BenchmarkWorker(cfg)
        self._worker.controller_done.connect(self._on_controller_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

    def _on_progress(self, text: str, pct: int):
        self._status_lbl.setText(text)
        if pct > 0:
            self._progress_bar.setValue(pct)

    def _on_controller_done(self, result: BenchmarkResult):
        self._add_summary_row(result)
        self._add_dist_row(result)
        self._colour_best_reward()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("Stopping…")

    def _on_finished(self, results, records):
        self._all_records = records
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._save_btn.setEnabled(bool(records))
        n = len(results)
        suffix = " (stopped early)" if self._worker and self._worker._stop.is_set() else ""
        self._status_lbl.setText(f"Done — {n} controller(s) benchmarked.{suffix}")

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "Benchmark error", msg)

    def _on_save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save benchmark CSV", "", "CSV (*.csv);;All (*)"
        )
        if path:
            write_csv(self._all_records, path)
            self._status_lbl.setText(f"Saved → {Path(path).name}")

    # ── Table helpers ──────────────────────────────────────────────────────────

    def _add_summary_row(self, r: BenchmarkResult):
        tbl = self._summary_table
        row = tbl.rowCount()
        tbl.insertRow(row)

        if r.pipeline_distribution:
            top_key = max(r.pipeline_distribution, key=r.pipeline_distribution.get)
            top_str = f"{top_key} ({r.pipeline_distribution[top_key] * 100:.0f}%)"
        else:
            top_str = "N/A"

        values = [
            r.controller_name,
            f"{r.mean_reward:.3f}",
            f"{r.std_reward:.3f}",
            f"{r.min_reward:.3f}",
            f"{r.max_reward:.3f}",
            f"{r.mean_latency_ms:.1f}",
            f"{r.p95_latency_ms:.1f}",
            str(r.total_switches),
            top_str,
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, col, item)

    def _add_dist_row(self, r: BenchmarkResult):
        tbl = self._dist_table
        row = tbl.rowCount()
        tbl.insertRow(row)
        tbl.setItem(row, 0, QTableWidgetItem(r.controller_name))
        for col, p_name in enumerate(_DIST_PIPELINES, start=1):
            pct = r.pipeline_distribution.get(p_name, 0.0) * 100
            item = QTableWidgetItem(f"{pct:.0f}%")
            item.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, col, item)

    def _colour_best_reward(self):
        """Highlight best mean reward green, worst red in the summary table."""
        tbl = self._summary_table
        n = tbl.rowCount()
        if n < 2:
            return
        rewards = []
        for row in range(n):
            item = tbl.item(row, 1)
            try:
                rewards.append(float(item.text()))
            except (ValueError, AttributeError):
                rewards.append(0.0)
        best = max(rewards)
        worst = min(rewards)
        for row, val in enumerate(rewards):
            item = tbl.item(row, 1)
            if item is None:
                continue
            if val == best:
                item.setBackground(QColor(40, 90, 50))
            elif val == worst:
                item.setBackground(QColor(90, 40, 40))
            else:
                item.setBackground(QColor(30, 30, 30))


# ── Validate workers ─────────────────────────────────────────────────────────

_VALIDATE_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_MODE_SINGLE     = 0
_MODE_PIPELINES  = 1
_MODE_CONTROLLERS = 2

_RUN_LABELS = ["VALIDATE MODEL", "COMPARE PIPELINES", "COMPARE CONTROLLERS"]


class ValidateWorker(QThread):
    """Unified worker for all three validate modes."""
    progress = pyqtSignal(str, int)   # (status_text, pct 0-100)
    item_done = pyqtSignal(object)    # partial result dict (pipeline/controller comparison)
    finished  = pyqtSignal(object)    # final result (dict for single model, list for compare)
    error     = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._run()
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _run(self):
        if self._stop.is_set():
            return
        mode = self.cfg["mode"]

        if mode == _MODE_SINGLE:
            self.progress.emit("Validating model… (this may take a while)", 10)
            result = run_validation(
                model_path=self.cfg["model_path"],
                data_yaml=self.cfg["data_yaml"],
                split=self.cfg["split"],
                conf=self.cfg["conf"],
                imgsz=self.cfg["imgsz"],
            )
            self.progress.emit("Done", 100)
            self.finished.emit({"mode": mode, "result": result})

        elif mode == _MODE_PIPELINES:
            selected = self.cfg["selected"]
            n_total  = len(selected)

            def cb(name, i, n):
                if self._stop.is_set():
                    raise InterruptedError("Stopped by user")
                pipe_idx = selected.index(name) if name in selected else 0
                pct = int((pipe_idx * n + i) / max(n_total * n, 1) * 100)
                self.progress.emit(f"Evaluating {name}… image {i + 1}/{n}", pct)

            results = run_pipeline_comparison(
                data_yaml=self.cfg["data_yaml"],
                split=self.cfg["split"],
                conf=self.cfg["conf"],
                model_path=self.cfg.get("model_path") or None,
                selected_pipelines=selected,
                imgsz=self.cfg.get("imgsz", 640),
                progress_cb=cb,
                result_cb=lambda r: self.item_done.emit(r),
            )
            self.progress.emit("Done", 100)
            self.finished.emit({"mode": mode, "result": results})

        elif mode == _MODE_CONTROLLERS:
            selected = self.cfg["selected"]
            n_total  = len(selected)

            def cb(name, i, n):
                if self._stop.is_set():
                    raise InterruptedError("Stopped by user")
                ctrl_idx = selected.index(name) if name in selected else 0
                pct = int((ctrl_idx * n + i) / max(n_total * n, 1) * 100)
                self.progress.emit(f"Evaluating {name}… image {i + 1}/{n}", pct)

            _replay_path: str | None = None
            if self.cfg.get("collect_replay"):
                from ..evaluation.replay_buffer import _DEFAULT_REPLAY_PATH
                _replay_path = _DEFAULT_REPLAY_PATH

            results = run_controller_comparison(
                data_yaml=self.cfg["data_yaml"],
                split=self.cfg["split"],
                conf=self.cfg["conf"],
                model_path=self.cfg.get("model_path") or None,
                selected_controllers=selected,
                imgsz=self.cfg.get("imgsz", 640),
                progress_cb=cb,
                result_cb=lambda r: self.item_done.emit(r),
                replay_path=_replay_path,
            )
            self.progress.emit("Done", 100)
            self.finished.emit({"mode": mode, "result": results})


# ── Validate widget ───────────────────────────────────────────────────────────

_VAL_OVERALL_COLS  = ["Metric", "Value"]
_VAL_CLASS_COLS    = ["Class", "AP50", "AP50-95", "Precision", "Recall"]
_VAL_COMPARE_COLS  = ["Name", "mAP50", "mAP50-95", "Precision", "Recall", "Latency (ms)", "Missed"]
_VAL_CTRL_COLS     = ["Controller", "mAP50", "mAP50-95", "Precision", "Recall", "Latency (ms)", "Missed", "Top Pipeline"]

_PROG_STYLE = (
    "QProgressBar { background: #3c3c3c; border: none; border-radius: 3px; height: 8px; }"
    "QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }"
)


def _styled_lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
    return lbl


class ValidateWidget(QWidget):
    """
    Validate tab — three modes selectable via radio buttons:

    Single Model   — runs YOLO.val() on the model selected in the main panel
    Pipelines      — runs each pipeline end-to-end on the dataset and compares mAP
    Controllers    — lets each controller select a pipeline per image and compares mAP
    """

    def __init__(self, model_getter=None):
        super().__init__()
        self._model_getter = model_getter or (lambda: "")
        self._worker: ValidateWorker | None = None
        self._last_report: dict | None = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel())

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        panel.setStyleSheet(_PANEL)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 14, 12, 14)
        v.setSpacing(0)

        # ── Dataset ──
        v.addWidget(_cap("DATASET  (data.yaml)"))
        v.addSpacing(4)
        ds_row = QHBoxLayout()
        ds_row.setSpacing(4)
        self._data_edit = QLineEdit()
        self._data_edit.setPlaceholderText("path/to/data.yaml")
        self._data_edit.setStyleSheet(_INPUT)
        ds_row.addWidget(self._data_edit)
        btn_ds = QPushButton("…")
        btn_ds.setFixedWidth(32)
        btn_ds.setStyleSheet(_BTN_SMALL)
        btn_ds.clicked.connect(self._on_browse_data)
        ds_row.addWidget(btn_ds)
        v.addLayout(ds_row)
        v.addSpacing(10)

        # ── Split ──
        v.addWidget(_divider())
        v.addSpacing(8)
        split_row = QHBoxLayout()
        split_row.addWidget(_cap("SPLIT"))
        split_row.addStretch()
        self._split_combo = QComboBox()
        self._split_combo.addItems(["val", "valid", "test", "train"])
        self._split_combo.setStyleSheet(_INPUT)
        self._split_combo.setFixedWidth(90)
        split_row.addWidget(self._split_combo)
        v.addLayout(split_row)
        v.addSpacing(10)

        # ── Confidence ──
        v.addWidget(_divider())
        v.addSpacing(8)
        self._conf_cap = _cap("CONFIDENCE  0.25")
        v.addWidget(self._conf_cap)
        v.addSpacing(4)
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(25)
        self._conf_slider.setStyleSheet(_SLIDER)
        self._conf_slider.valueChanged.connect(
            lambda val: self._conf_cap.setText(f"CONFIDENCE  {val / 100:.2f}")
        )
        v.addWidget(self._conf_slider)
        v.addSpacing(10)

        # ── Image size ──
        v.addWidget(_divider())
        v.addSpacing(8)
        sz_row = QHBoxLayout()
        sz_row.addWidget(_cap("IMAGE SIZE"))
        sz_row.addStretch()
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(32, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        self._imgsz_spin.setStyleSheet(_INPUT)
        self._imgsz_spin.setFixedWidth(70)
        sz_row.addWidget(self._imgsz_spin)
        v.addLayout(sz_row)
        v.addSpacing(10)

        # ── Mode ──
        v.addWidget(_divider())
        v.addSpacing(8)
        v.addWidget(_cap("MODE"))
        v.addSpacing(4)
        self._rb_single = QRadioButton("Single Model")
        self._rb_pipelines = QRadioButton("Pipelines")
        self._rb_controllers = QRadioButton("Controllers")
        self._rb_single.setChecked(True)
        for rb in (self._rb_single, self._rb_pipelines, self._rb_controllers):
            rb.setStyleSheet(_RADIO)
            v.addWidget(rb)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._rb_single,      _MODE_SINGLE)
        self._mode_grp.addButton(self._rb_pipelines,   _MODE_PIPELINES)
        self._mode_grp.addButton(self._rb_controllers, _MODE_CONTROLLERS)
        self._mode_grp.buttonClicked.connect(self._on_mode_changed)
        v.addSpacing(8)

        # ── Mode-specific options (stacked) ──
        self._opts_stack = QStackedWidget()
        self._opts_stack.addWidget(self._build_opts_single())
        self._opts_stack.addWidget(self._build_opts_pipelines())
        self._opts_stack.addWidget(self._build_opts_controllers())
        v.addWidget(self._opts_stack)
        v.addSpacing(10)

        # ── Buttons ──
        v.addWidget(_divider())
        v.addSpacing(8)
        self._run_btn = QPushButton(_RUN_LABELS[_MODE_SINGLE])
        self._run_btn.setStyleSheet(_BTN_START)
        self._run_btn.clicked.connect(self._on_run)
        v.addWidget(self._run_btn)
        v.addSpacing(5)
        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setStyleSheet(_BTN_STOP)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        v.addWidget(self._stop_btn)
        v.addSpacing(5)
        self._save_btn = QPushButton("SAVE REPORT")
        self._save_btn.setStyleSheet(_BTN_SMALL)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_report)
        v.addWidget(self._save_btn)
        v.addSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(_PROG_STYLE)
        v.addWidget(self._progress_bar)
        v.addSpacing(4)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        v.addWidget(self._status_lbl)

        v.addStretch()
        return panel

    def _build_opts_single(self) -> QWidget:
        """No extra controls for single-model mode."""
        w = QWidget()
        lbl = QLabel("Model path taken from\nthe main panel MODEL field.")
        lbl.setStyleSheet("color: #666; font-size: 10px;")
        lbl.setWordWrap(True)
        QVBoxLayout(w).addWidget(lbl)
        return w

    def _build_opts_pipelines(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(_cap("PIPELINES"))
        v.addSpacing(2)
        self._pipe_chks: dict[str, QCheckBox] = {}
        defaults = {"fast_baseline": True, "clahe_pipeline": True,
                    "tiled": True, "high_res": False,
                    "bright_pipeline": True, "denoise_pipeline": True}
        for name, checked in defaults.items():
            chk = QCheckBox(name)
            chk.setChecked(checked)
            chk.setStyleSheet(_CHECK)
            v.addWidget(chk)
            self._pipe_chks[name] = chk
        return w

    def _build_opts_controllers(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(_cap("CONTROLLERS"))
        v.addSpacing(2)
        self._ctrl_chks: dict[str, QCheckBox] = {}
        defaults = {"none": False, "rule": True, "ucb": True, "contextual": True,
                    "decision_tree": True, "random_forest": True,
                    "neural_net": True, "neural_rl": True}
        for name, checked in defaults.items():
            chk = QCheckBox(name)
            chk.setChecked(checked)
            chk.setStyleSheet(_CHECK)
            v.addWidget(chk)
            self._ctrl_chks[name] = chk
        v.addSpacing(8)
        v.addWidget(_cap("LEARNING"))
        v.addSpacing(2)
        self._chk_ctrl_replay = QCheckBox("Collect replay data")
        self._chk_ctrl_replay.setToolTip(
            "After evaluation, export ground-truth recall rewards to\n"
            "replay_buffer.jsonl so the Train tab can retrain controllers\n"
            "on real labelled data."
        )
        self._chk_ctrl_replay.setChecked(True)
        self._chk_ctrl_replay.setStyleSheet(_CHECK)
        v.addWidget(self._chk_ctrl_replay)
        return w

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        right.setStyleSheet("background-color: #1a1a1a;")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 10)
        rv.setSpacing(6)

        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(self._build_single_results())
        self._right_stack.addWidget(self._build_compare_results("Pipeline"))
        self._right_stack.addWidget(self._build_compare_results("Controller"))
        rv.addWidget(self._right_stack)
        return right

    def _build_single_results(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(_styled_lbl("Overall Metrics"))
        self._overall_table = QTableWidget(4, 2)
        self._overall_table.setHorizontalHeaderLabels(_VAL_OVERALL_COLS)
        self._overall_table.setStyleSheet(_TBL_STYLE)
        self._overall_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overall_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._overall_table.horizontalHeader().setStretchLastSection(True)
        self._overall_table.verticalHeader().setVisible(False)
        self._overall_table.setMaximumHeight(130)
        for row, name in enumerate(["mAP50", "mAP50-95", "Precision", "Recall"]):
            it = QTableWidgetItem(name)
            it.setTextAlignment(Qt.AlignCenter)
            self._overall_table.setItem(row, 0, it)
            vit = QTableWidgetItem("—")
            vit.setTextAlignment(Qt.AlignCenter)
            self._overall_table.setItem(row, 1, vit)
        v.addWidget(self._overall_table)
        v.addWidget(_styled_lbl("Per-Class Results"))
        self._class_table = QTableWidget(0, len(_VAL_CLASS_COLS))
        self._class_table.setHorizontalHeaderLabels(_VAL_CLASS_COLS)
        self._class_table.setStyleSheet(_TBL_STYLE)
        self._class_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._class_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._class_table.horizontalHeader().setStretchLastSection(True)
        self._class_table.verticalHeader().setVisible(False)
        v.addWidget(self._class_table, stretch=1)
        self._speed_lbl = QLabel("")
        self._speed_lbl.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        v.addWidget(self._speed_lbl)
        return w

    def _build_compare_results(self, kind: str) -> QWidget:
        """Shared layout for pipeline and controller comparison tables."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        cols = _VAL_CTRL_COLS if kind == "Controller" else _VAL_COMPARE_COLS
        v.addWidget(_styled_lbl(f"{kind} Comparison"))
        tbl = QTableWidget(0, len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.setStyleSheet(_TBL_STYLE)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.verticalHeader().setVisible(False)
        v.addWidget(tbl, stretch=1)
        if kind == "Pipeline":
            self._pipe_cmp_table = tbl
        else:
            self._ctrl_cmp_table = tbl
        return w

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, btn):
        mode = self._mode_grp.id(btn)
        self._opts_stack.setCurrentIndex(mode)
        self._right_stack.setCurrentIndex(mode)
        self._run_btn.setText(_RUN_LABELS[mode])
        self._save_btn.setEnabled(False)

    def _on_browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self._data_edit.setText(path)

    def _current_mode(self) -> int:
        return self._mode_grp.checkedId()

    def _on_run(self):
        data_yaml = self._data_edit.text().strip()
        if not data_yaml:
            QMessageBox.warning(self, "No dataset", "Select a data.yaml file first.")
            return

        mode = self._current_mode()
        model_path = self._model_getter().strip() or str(_VALIDATE_MODELS_DIR / "yolov8n.pt")

        cfg: dict = {
            "mode":       mode,
            "data_yaml":  data_yaml,
            "split":      self._split_combo.currentText(),
            "conf":       self._conf_slider.value() / 100.0,
            "imgsz":      self._imgsz_spin.value(),
            "model_path": model_path,
        }

        if mode == _MODE_SINGLE:
            # reset single-model display
            for row in range(self._overall_table.rowCount()):
                self._overall_table.item(row, 1).setText("—")
            self._class_table.setRowCount(0)
            self._speed_lbl.setText("")
        elif mode == _MODE_PIPELINES:
            selected = [n for n, c in self._pipe_chks.items() if c.isChecked()]
            if not selected:
                QMessageBox.warning(self, "No pipelines", "Select at least one pipeline.")
                return
            cfg["selected"] = selected
            self._pipe_cmp_table.setRowCount(0)
        elif mode == _MODE_CONTROLLERS:
            selected = [n for n, c in self._ctrl_chks.items() if c.isChecked()]
            if not selected:
                QMessageBox.warning(self, "No controllers", "Select at least one controller.")
                return
            cfg["selected"] = selected
            cfg["collect_replay"] = self._chk_ctrl_replay.isChecked()
            self._ctrl_cmp_table.setRowCount(0)

        self._last_report = None
        self._save_btn.setEnabled(False)
        self._worker = ValidateWorker(cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._status_lbl.setText("Starting…")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("Stopping…")

    def _on_progress(self, text: str, pct: int):
        self._status_lbl.setText(text)
        self._progress_bar.setValue(pct)

    def _on_item_done(self, result: dict):
        """Populate a row as each pipeline/controller finishes."""
        if "pipeline_name" in result:
            self._add_compare_row(self._pipe_cmp_table, result, kind="pipeline")
            self._colour_best(self._pipe_cmp_table, col=1)
        elif "controller_name" in result:
            self._add_compare_row(self._ctrl_cmp_table, result, kind="controller")
            self._colour_best(self._ctrl_cmp_table, col=1)

    def _on_finished(self, payload: dict):
        mode   = payload["mode"]
        result = payload["result"]
        self._last_report = payload
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._save_btn.setEnabled(True)

        if mode == _MODE_SINGLE:
            vals = [result["map50"], result["map"], result["precision"], result["recall"]]
            for row, v in enumerate(vals):
                self._overall_table.item(row, 1).setText(f"{v:.4f}  ({v * 100:.1f}%)")
            self._class_table.setRowCount(0)
            per_class = result.get("per_class", [])
            best_ap50 = max((c["ap50"] for c in per_class), default=0.0)
            for cls in per_class:
                row = self._class_table.rowCount()
                self._class_table.insertRow(row)
                for col, val in enumerate([
                    cls["class"],
                    f"{cls['ap50']:.4f}", f"{cls['ap']:.4f}",
                    f"{cls['precision']:.4f}", f"{cls['recall']:.4f}",
                ]):
                    it = QTableWidgetItem(val)
                    it.setTextAlignment(Qt.AlignCenter)
                    self._class_table.setItem(row, col, it)
                if cls["ap50"] == best_ap50 and len(per_class) > 1:
                    for col in range(len(_VAL_CLASS_COLS)):
                        it = self._class_table.item(row, col)
                        if it:
                            it.setBackground(QColor(40, 90, 50))
            spd = result.get("speed", {})
            self._speed_lbl.setText(
                f"Preprocess: {spd.get('preprocess', 0):.1f} ms  |  "
                f"Inference: {spd.get('inference', 0):.1f} ms  |  "
                f"Postprocess: {spd.get('postprocess', 0):.1f} ms"
            )
            self._status_lbl.setText(f"Done — {result.get('n_classes', 0)} class(es).")

        elif mode == _MODE_PIPELINES:
            self._status_lbl.setText(f"Done — {len(result)} pipeline(s) evaluated.")

        elif mode == _MODE_CONTROLLERS:
            self._status_lbl.setText(f"Done — {len(result)} controller(s) evaluated.")

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "Evaluation error", msg)

    def _on_save_report(self):
        if not self._last_report:
            return
        import json
        path, _ = QFileDialog.getSaveFileName(
            self, "Save report", "", "JSON (*.json);;All (*)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self._last_report, f, indent=2, default=str)
            self._status_lbl.setText(f"Saved -> {Path(path).name}")

    # ── Table helpers ──────────────────────────────────────────────────────────

    def _add_compare_row(self, tbl: QTableWidget, r: dict, kind: str):
        row = tbl.rowCount()
        tbl.insertRow(row)
        if kind == "pipeline":
            name = r.get("pipeline_name", "?")
            dist = ""
        else:
            name = r.get("controller_name", "?")
            dist_d = r.get("pipeline_distribution", {})
            if dist_d:
                top = max(dist_d, key=dist_d.get)
                dist = f"{top} ({dist_d[top]*100:.0f}%)"
            else:
                dist = "—"

        fn  = r.get("total_fn")
        tp  = r.get("total_tp")
        gt  = (tp or 0) + (fn or 0)
        missed_str = f"{fn} / {gt}" if fn is not None else "—"

        values = [
            name,
            f"{r.get('map50', 0):.4f}",
            f"{r.get('map', 0):.4f}",
            f"{r.get('precision', 0):.4f}",
            f"{r.get('recall', 0):.4f}",
            f"{r.get('mean_latency_ms', 0):.1f}",
            missed_str,
        ]
        if kind == "controller":
            values.append(dist)

        for col, val in enumerate(values):
            it = QTableWidgetItem(val)
            it.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, col, it)

    def _colour_best(self, tbl: QTableWidget, col: int = 1):
        """Highlight best mAP50 row green, worst red."""
        n = tbl.rowCount()
        if n < 2:
            return
        vals = []
        for row in range(n):
            try:
                vals.append(float(tbl.item(row, col).text()))
            except (ValueError, AttributeError):
                vals.append(0.0)
        best, worst = max(vals), min(vals)
        for row, v in enumerate(vals):
            bg = QColor(40, 90, 50) if v == best else (
                 QColor(90, 40, 40) if v == worst else QColor(30, 30, 30))
            for c in range(tbl.columnCount()):
                it = tbl.item(row, c)
                if it:
                    it.setBackground(bg)


# ── Video display widget ──────────────────────────────────────────────────────

class VideoLabel(QLabel):
    """
    Video/image display with interactive pan & zoom.

    Controls
    --------
    Scroll wheel        — zoom in / out (centred on cursor)
    Left-button drag    — pan when zoomed in
    Double-click        — reset zoom to fit-to-window
    """

    def __init__(self):
        super().__init__("Configure settings on the left, then press  START")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #111111; color: #555555; font-size: 14px;")
        self.setMouseTracking(True)

        self._raw_frame: np.ndarray | None = None
        self._zoom: float = 1.0          # 1.0 = fit to widget
        self._pan: list[float] = [0.5, 0.5]   # normalised centre (0-1)
        self._drag_start: tuple | None = None  # (QPoint, [pan_x, pan_y])

    # ── public ────────────────────────────────────────────────────────────

    def update_frame(self, img: np.ndarray) -> None:
        self._raw_frame = img
        self._render()

    # ── internal rendering ────────────────────────────────────────────────

    def _render(self) -> None:
        if self._raw_frame is None:
            return
        img = self._raw_frame
        fh, fw = img.shape[:2]

        if self._zoom <= 1.0:
            crop = img
        else:
            # Viewport size in original image pixels
            vw = fw / self._zoom
            vh = fh / self._zoom
            # Top-left corner of viewport
            cx, cy = self._pan[0] * fw, self._pan[1] * fh
            x1 = int(max(0, round(cx - vw / 2)))
            y1 = int(max(0, round(cy - vh / 2)))
            x2 = int(min(fw, x1 + round(vw)))
            y2 = int(min(fh, y1 + round(vh)))
            # Clamp so viewport doesn't go out of bounds
            if x2 - x1 < round(vw): x1 = max(0, x2 - int(round(vw)))
            if y2 - y1 < round(vh): y1 = max(0, y2 - int(round(vh)))
            crop = img[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch2, cw2 = rgb.shape[:2]
        qt_img = QImage(rgb.data.tobytes(), cw2, ch2, 3 * cw2, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)

        # Zoom level overlay
        if self._zoom > 1.005:
            from PyQt5.QtGui import QPainter, QFont
            from PyQt5.QtCore import QRect
            p = QPainter(pixmap)
            p.setOpacity(0.75)
            p.fillRect(QRect(pixmap.width() - 74, 6, 68, 22),
                       QColor(0, 0, 0))
            p.setOpacity(1.0)
            p.setPen(QColor(0, 220, 100))
            font = QFont("Consolas", 10, QFont.Bold)
            p.setFont(font)
            p.drawText(QRect(pixmap.width() - 74, 6, 68, 22),
                       Qt.AlignCenter, f"🔍 {self._zoom:.1f}×")
            p.end()
            self.setPixmap(pixmap)

    def _display_rect(self, fw: int, fh: int):
        """Return (x_off, y_off, disp_w, disp_h) of the image within this widget."""
        lw, lh = self.width(), self.height()
        scale = min(lw / max(fw, 1), lh / max(fh, 1))
        dw, dh = fw * scale, fh * scale
        return (lw - dw) / 2.0, (lh - dh) / 2.0, dw, dh

    def _widget_to_img_norm(self, wx: float, wy: float) -> tuple[float, float]:
        """Map a widget pixel (wx, wy) to a normalised image coordinate (0-1, 0-1)."""
        if self._raw_frame is None:
            return 0.5, 0.5
        fh, fw = self._raw_frame.shape[:2]
        x_off, y_off, dw, dh = self._display_rect(fw, fh)
        # Fraction within the displayed image area
        frac_x = (wx - x_off) / max(dw, 1)
        frac_y = (wy - y_off) / max(dh, 1)
        # Map to original image coords via pan/zoom
        nx = self._pan[0] + (frac_x - 0.5) / self._zoom
        ny = self._pan[1] + (frac_y - 0.5) / self._zoom
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _clamp_pan(self) -> None:
        hw = 0.5 / self._zoom
        self._pan[0] = max(hw, min(1.0 - hw, self._pan[0]))
        self._pan[1] = max(hw, min(1.0 - hw, self._pan[1]))

    # ── events ────────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.18 if delta > 0 else (1.0 / 1.18)

        # Image point under the cursor stays fixed after zoom
        mx, my = event.pos().x(), event.pos().y()
        img_nx, img_ny = self._widget_to_img_norm(mx, my)

        self._zoom = max(1.0, min(12.0, self._zoom * factor))

        if self._zoom > 1.0:
            # Adjust pan so cursor-image point is preserved
            if self._raw_frame is not None:
                fh, fw = self._raw_frame.shape[:2]
                x_off, y_off, dw, dh = self._display_rect(fw, fh)
                frac_x = (mx - x_off) / max(dw, 1)
                frac_y = (my - y_off) / max(dh, 1)
                self._pan[0] = img_nx - (frac_x - 0.5) / self._zoom
                self._pan[1] = img_ny - (frac_y - 0.5) / self._zoom
            self._clamp_pan()
        else:
            self._pan = [0.5, 0.5]

        self._render()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._zoom > 1.0:
            self._drag_start = (event.pos(), list(self._pan))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start and self._raw_frame is not None:
            start_pos, start_pan = self._drag_start
            fh, fw = self._raw_frame.shape[:2]
            _, _, dw, dh = self._display_rect(fw, fh)
            # Drag offsets as fractions of the displayed image
            ddx = (event.pos().x() - start_pos.x()) / max(dw, 1)
            ddy = (event.pos().y() - start_pos.y()) / max(dh, 1)
            self._pan[0] = start_pan[0] - ddx / self._zoom
            self._pan[1] = start_pan[1] - ddy / self._zoom
            self._clamp_pan()
            self._render()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click resets to fit-to-window."""
        self._zoom = 1.0
        self._pan = [0.5, 0.5]
        self._render()
        super().mouseDoubleClickEvent(event)


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

        # ── Right side: tabs (Live preview + Benchmark) ───────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border: none; background: #111; }"
            "QTabBar::tab { background: #2b2b2b; color: #aaa; padding: 7px 18px;"
            "  border: none; font-size: 12px; }"
            "QTabBar::tab:selected { background: #1a1a1a; color: #fff; }"
            "QTabBar::tab:hover { background: #333; }"
        )
        tabs.addTab(self._build_video_area(), "▶  Live")
        tabs.addTab(
            BenchmarkWidget(
                source_getter=lambda: self.source_edit.text().strip(),
                model_getter=lambda: self._model_edit.text().strip(),
            ),
            "📊  Benchmark",
        )
        tabs.addTab(
            ValidateWidget(model_getter=lambda: self._model_edit.text().strip()),
            "🎯  Validate",
        )

        _replay_default = str(Path(__file__).resolve().parent.parent.parent / "replay_buffer.jsonl")
        self._train_widget = TrainWidget(replay_path=_replay_default)
        self._train_widget.models_updated.connect(self._on_models_updated)
        tabs.addTab(self._train_widget, "🔁  Train")

        root.addWidget(tabs)

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
        v.addSpacing(10)

        # ── Model ──
        v.addWidget(_cap("MODEL  (optional)"))
        v.addSpacing(4)
        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("Default (models/yolov8n.pt)")
        self._model_edit.setStyleSheet(_INPUT)
        model_row.addWidget(self._model_edit)
        btn_model = QPushButton("…")
        btn_model.setFixedWidth(32)
        btn_model.setStyleSheet(_BTN_SMALL)
        btn_model.clicked.connect(self._on_browse_model)
        model_row.addWidget(btn_model)
        v.addLayout(model_row)
        v.addSpacing(14)

        # ── Controller ──
        v.addWidget(_divider())
        v.addSpacing(10)
        v.addWidget(_cap("CONTROLLER"))
        v.addSpacing(4)
        self.ctrl_combo = QComboBox()
        self.ctrl_combo.addItems([
            "none",
            "rule", "ucb", "contextual",
            "decision_tree", "random_forest",
            "neural_net", "neural_rl",
        ])
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

        # ── Playback controls ──
        v.addSpacing(8)
        playback_row = QHBoxLayout()
        playback_row.setSpacing(4)
        self.step_back_btn = QPushButton("◀")
        self.step_back_btn.setFixedWidth(36)
        self.step_back_btn.setEnabled(False)
        self.step_back_btn.clicked.connect(lambda: self._worker.step(-1) if self._worker else None)
        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_toggle)
        self.step_fwd_btn = QPushButton("▶")
        self.step_fwd_btn.setFixedWidth(36)
        self.step_fwd_btn.setEnabled(False)
        self.step_fwd_btn.clicked.connect(lambda: self._worker.step(1) if self._worker else None)
        playback_row.addWidget(self.step_back_btn)
        playback_row.addWidget(self.pause_btn)
        playback_row.addWidget(self.step_fwd_btn)
        v.addLayout(playback_row)
        v.addSpacing(4)
        self._frame_lbl = QLabel("Frame: — / —")
        self._frame_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self._frame_lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(self._frame_lbl)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        v.addWidget(self.seek_slider)

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

    def _on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", "",
            "Model files (*.pt *.onnx);;All files (*)",
        )
        if path:
            self._model_edit.setText(path)

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
            "model_path": self._model_edit.text().strip(),
        }


        self._pause_active = False
        self._worker = PipelineWorker(cfg)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.paused.connect(self._on_paused)
        self._worker.frame_info.connect(self._on_frame_info)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ PAUSE")
        self.step_back_btn.setEnabled(False)
        self.step_fwd_btn.setEnabled(False)
        self._frame_lbl.setText("Frame: — / —")
        self.seek_slider.setMaximum(0)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(True)
        self.video_label.setText("Loading model…")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_pause_toggle(self):
        if not self._worker:
            return
        if self._pause_active:
            self._worker.resume()
        else:
            self._worker.pause()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_paused(self, is_paused: bool):
        self._pause_active = is_paused
        self.pause_btn.setText("▶ RESUME" if is_paused else "⏸ PAUSE")
        self.step_back_btn.setEnabled(is_paused)
        self.step_fwd_btn.setEnabled(is_paused)

    def _on_slider_moved(self, value: int):
        if self._worker:
            self._worker.seek(value)

    def _on_frame_info(self, idx: int, total: int):
        self._frame_lbl.setText(f"Frame: {idx + 1} / {total}")
        if self.seek_slider.maximum() != total - 1:
            self.seek_slider.setMaximum(max(0, total - 1))
        self.seek_slider.setValue(idx)

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
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ PAUSE")
        self.step_back_btn.setEnabled(False)
        self.step_fwd_btn.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self._pause_active = False
        self.statusBar().showMessage("Run complete.", 4000)

    def _on_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.step_back_btn.setEnabled(False)
        self.step_fwd_btn.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self._pause_active = False
        QMessageBox.critical(self, "Pipeline error", msg)

    def _on_models_updated(self) -> None:
        """
        Called when the Train tab reloads models.

        If a live-inference worker is currently running, its controller is hot-
        reloaded in-place so the new weights take effect without restarting.
        Otherwise, a status-bar message confirms the reload — the next run will
        automatically pick up the freshly saved model files.
        """
        reloaded: list[str] = []
        if self._worker is not None and hasattr(self._worker, "_orchestrator"):
            ctrl = getattr(self._worker._orchestrator, "_controller", None)
            if ctrl is not None and hasattr(ctrl, "reload"):
                ctrl.reload()
                reloaded.append(type(ctrl).__name__)

        if reloaded:
            self.statusBar().showMessage(
                f"Models reloaded live: {', '.join(reloaded)}", 5000
            )
        else:
            self.statusBar().showMessage(
                "Models reloaded — next run will use updated weights.", 5000
            )


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
