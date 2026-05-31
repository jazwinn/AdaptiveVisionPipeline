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
from ..controller.neural_net import NeuralNetController
from ..controller.neural_rl import NeuralRLController
from ..controller.none import NoneController
from ..controller.orchestrator import PipelineOrchestrator
from ..evaluation.metrics import WindowMetrics, compute_reward
from ..evaluation.replay_buffer import ReplayBuffer
from ..evaluation.benchmark import (
    BenchmarkResult, PerWindowRecord,
    build_pipelines, build_controllers, run_controller, write_csv,
)
from ..experiments.logger import ExperimentLogger

import cv2
import numpy as np
import supervision as sv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QRadioButton, QSizePolicy, QSlider, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
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

        model_path = cfg.get("model_path") or None
        # None triggers each pipeline's built-in default (models/ at project root)
        pipelines = [PipelineA(conf=conf, model_path=model_path)]
        if not cfg["fast_only"]:
            pipelines.append(PipelineD(conf=conf, model_path=model_path))
            pipelines.append(PipelineC(conf=conf, model_path=model_path))
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

_DIST_PIPELINES = ["fast_baseline", "clahe_pipeline", "tiled", "high_res"]

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
