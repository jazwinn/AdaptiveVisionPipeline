"""
Train tab widget — trigger offline retraining of NN / RL / DT controllers
from within the GUI, monitor progress, and hot-reload trained models without
restarting the application.

Closed-loop flow
----------------
1. Collect real data   → run Live inference or Validate with "Append replay buffer" on
2. Train               → click a button here; subprocess streams output to the log
3. Reload              → click "Reload All Models"; controllers swap weights in-place
4. Validate / Live     → improved routing visible immediately in the next run
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# ── Paths (resolved once at import time) ─────────────────────────────────────

_MODELS_DIR   = Path(__file__).resolve().parent.parent / "controller" / "models"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MODEL_FILES: dict[str, Path] = {
    "nn": _MODELS_DIR / "nn_controller.pt",
    "rl": _MODELS_DIR / "rl_controller.pt",
    "dt": _MODELS_DIR / "dt_controller.joblib",
    "rf": _MODELS_DIR / "rf_controller.joblib",
}

_TRAIN_CMDS: dict[str, list[str]] = {
    "nn": ["-m", "Source.controller.train_nn"],
    "rl": ["-m", "Source.controller.train_rl"],
    "dt": ["-m", "Source.controller.train_dt", "--model", "both"],
}

# ── Styles ────────────────────────────────────────────────────────────────────

_BTN = (
    "QPushButton { background: #3c3c3c; color: #e8e8e8; border: none;"
    "  border-radius: 4px; padding: 6px 14px; font-size: 12px; }"
    "QPushButton:hover { background: #4a4a4a; }"
    "QPushButton:disabled { background: #2e2e2e; color: #666; }"
    "QPushButton:pressed { background: #555; }"
)
_BTN_PRIMARY = (
    "QPushButton { background: #4a7fc1; color: #fff; border: none;"
    "  border-radius: 4px; padding: 6px 14px; font-size: 12px; }"
    "QPushButton:hover { background: #5a8fd1; }"
    "QPushButton:disabled { background: #2e2e2e; color: #666; }"
)
_BTN_SUCCESS = (
    "QPushButton { background: #3a7a3a; color: #fff; border: none;"
    "  border-radius: 4px; padding: 6px 14px; font-size: 12px; }"
    "QPushButton:hover { background: #4a8a4a; }"
    "QPushButton:disabled { background: #2e2e2e; color: #666; }"
)
_GROUP = (
    "QGroupBox { color: #aaa; font-size: 11px; border: 1px solid #404040;"
    "  border-radius: 4px; margin-top: 8px; padding-top: 14px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
)
_LOG = (
    "QPlainTextEdit { background: #1a1a1a; color: #c8c8c8;"
    "  font-family: Consolas, monospace; font-size: 11px;"
    "  border: 1px solid #333; border-radius: 3px; }"
)
_PROG = (
    "QProgressBar { background: #3c3c3c; border: none; border-radius: 3px; height: 6px; }"
    "QProgressBar::chunk { background: #4a7fc1; border-radius: 3px; }"
)


# ── Worker thread ─────────────────────────────────────────────────────────────

class TrainWorker(QThread):
    """
    Runs one or more training scripts as subprocesses and streams stdout/stderr
    line-by-line via ``log_line``.  Emits ``finished(ok, target)`` when done.
    """
    log_line = pyqtSignal(str)          # one stdout/stderr line
    finished = pyqtSignal(bool, str)    # (success, target: "nn"/"rl"/"dt"/"all")

    def __init__(self, target: str, replay_path: str) -> None:
        super().__init__()
        self._target = target
        self._replay = replay_path

    def run(self) -> None:
        targets = ["nn", "rl", "dt"] if self._target == "all" else [self._target]
        ok = True
        for t in targets:
            self.log_line.emit(f"\n{'=' * 55}")
            self.log_line.emit(f"  Training: {t.upper()}")
            self.log_line.emit(f"{'=' * 55}")
            cmd = [sys.executable] + _TRAIN_CMDS[t] + ["--replay", self._replay]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(_PROJECT_ROOT),
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_line.emit(line.rstrip())
                rc = proc.wait()
                if rc != 0:
                    self.log_line.emit(f"[ERROR] {t.upper()} exited with code {rc}")
                    ok = False
                else:
                    self.log_line.emit(f"[OK] {t.upper()} training complete.")
            except Exception as exc:
                self.log_line.emit(f"[ERROR] Failed to launch {t}: {exc}")
                ok = False
        self.finished.emit(ok, self._target)


# ── Train tab widget ──────────────────────────────────────────────────────────

class TrainWidget(QWidget):
    """
    Self-contained tab for the closed-loop training workflow.

    Signals
    -------
    models_updated : emitted after "Reload All Models" so the app can
                     refresh any live controller references.
    """
    models_updated = pyqtSignal()

    def __init__(self, replay_path: str) -> None:
        super().__init__()
        self._replay_path = replay_path
        self._worker: TrainWorker | None = None
        self.setStyleSheet("background-color: #1e1e1e; color: #e8e8e8;")
        self._build_ui()
        self._refresh_buffer_stats()
        self._refresh_model_status()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        root.addWidget(self._build_buffer_group())
        root.addWidget(self._build_train_group())
        root.addWidget(self._build_deploy_group())
        root.addStretch()

    def _build_buffer_group(self) -> QGroupBox:
        grp = QGroupBox("Replay Buffer")
        grp.setStyleSheet(_GROUP)
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        row = QHBoxLayout()
        self._buf_entries_lbl = QLabel("Entries: —")
        self._buf_entries_lbl.setStyleSheet("color: #e8e8e8; font-size: 12px;")
        self._buf_last_lbl = QLabel("Last entry: —")
        self._buf_last_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setStyleSheet(_BTN)
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._refresh_buffer_stats)
        row.addWidget(self._buf_entries_lbl)
        row.addWidget(self._buf_last_lbl)
        row.addStretch()
        row.addWidget(refresh_btn)
        v.addLayout(row)

        path_lbl = QLabel(f"Path: {self._replay_path}")
        path_lbl.setStyleSheet("color: #666; font-size: 10px;")
        path_lbl.setWordWrap(True)
        v.addWidget(path_lbl)

        return grp

    def _build_train_group(self) -> QGroupBox:
        grp = QGroupBox("Train")
        grp.setStyleSheet(_GROUP)
        v = QVBoxLayout(grp)
        v.setSpacing(8)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_nn  = QPushButton("🧠 Train NN")
        self._btn_rl  = QPushButton("🎮 Train RL")
        self._btn_dt  = QPushButton("🌳 Train DT + RF")
        self._btn_all = QPushButton("🔄 Train All")

        for btn in (self._btn_nn, self._btn_rl, self._btn_dt):
            btn.setStyleSheet(_BTN)
        self._btn_all.setStyleSheet(_BTN_PRIMARY)

        self._btn_nn.clicked.connect(lambda: self._start_training("nn"))
        self._btn_rl.clicked.connect(lambda: self._start_training("rl"))
        self._btn_dt.clicked.connect(lambda: self._start_training("dt"))
        self._btn_all.clicked.connect(lambda: self._start_training("all"))

        btn_row.addWidget(self._btn_nn)
        btn_row.addWidget(self._btn_rl)
        btn_row.addWidget(self._btn_dt)
        btn_row.addWidget(self._btn_all)
        btn_row.addStretch()
        v.addLayout(btn_row)

        # Log output
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(_LOG)
        self._log.setMinimumHeight(220)
        self._log.setPlaceholderText(
            "Training output will appear here.\n\n"
            "Tip: run Validate → Controller Comparison with 'Append replay buffer'\n"
            "enabled first to collect ground-truth rewarded data, then train."
        )
        v.addWidget(self._log)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setStyleSheet(_PROG)
        self._progress.setRange(0, 0)   # indeterminate by default
        self._progress.setValue(0)
        self._progress.setVisible(False)
        v.addWidget(self._progress)

        # Status label
        self._train_status = QLabel("")
        self._train_status.setStyleSheet("color: #aaa; font-size: 11px;")
        v.addWidget(self._train_status)

        return grp

    def _build_deploy_group(self) -> QGroupBox:
        grp = QGroupBox("Deploy")
        grp.setStyleSheet(_GROUP)
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        reload_btn = QPushButton("✅ Reload All Models")
        reload_btn.setStyleSheet(_BTN_SUCCESS)
        reload_btn.setFixedWidth(180)
        reload_btn.clicked.connect(self._on_reload_models)
        v.addWidget(reload_btn)

        # Per-model status labels
        self._model_labels: dict[str, QLabel] = {}
        for key, path in _MODEL_FILES.items():
            lbl = QLabel(self._model_status_text(key, path))
            lbl.setStyleSheet("color: #aaa; font-size: 11px; padding-left: 4px;")
            v.addWidget(lbl)
            self._model_labels[key] = lbl

        return grp

    # ── Logic ─────────────────────────────────────────────────────────────

    def _refresh_buffer_stats(self) -> None:
        p = Path(self._replay_path)
        if not p.exists():
            self._buf_entries_lbl.setText("Entries: 0  (file not found)")
            self._buf_last_lbl.setText("")
            return
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            count = sum(1 for ln in lines if ln.strip())
            self._buf_entries_lbl.setText(f"Entries: {count:,}")
            # Find last valid timestamp
            last_ts: float | None = None
            import json
            for ln in reversed(lines):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    last_ts = rec.get("timestamp")
                    break
                except Exception:
                    continue
            if last_ts:
                ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ts))
                self._buf_last_lbl.setText(f"Last entry: {ts_str}")
            else:
                self._buf_last_lbl.setText("")
        except Exception as exc:
            self._buf_entries_lbl.setText(f"Entries: error ({exc})")
            self._buf_last_lbl.setText("")

    def _refresh_model_status(self) -> None:
        for key, path in _MODEL_FILES.items():
            lbl = self._model_labels.get(key)
            if lbl:
                lbl.setText(self._model_status_text(key, path))

    @staticmethod
    def _model_status_text(key: str, path: Path) -> str:
        name = key.upper()
        if not path.exists():
            return f"  {name}: ✗ not trained yet"
        mtime = path.stat().st_mtime
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        return f"  {name}: ✓ {ts}"

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self._btn_nn, self._btn_rl, self._btn_dt, self._btn_all):
            btn.setEnabled(enabled)

    def _start_training(self, target: str) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._log.clear()
        self._set_buttons_enabled(False)
        self._progress.setRange(0, 0)   # indeterminate pulse
        self._progress.setVisible(True)
        self._train_status.setText(f"Training {target.upper()}…")

        self._worker = TrainWorker(target=target, replay_path=self._replay_path)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_train_finished)
        self._worker.start()

    def _on_log_line(self, line: str) -> None:
        self._log.appendPlainText(line)
        # Auto-scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_train_finished(self, ok: bool, target: str) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._set_buttons_enabled(True)
        if ok:
            self._train_status.setText(f"✅ Training complete — click 'Reload All Models' to apply.")
            self._log.appendPlainText("\n✅ All done! New model files written to Source/controller/models/")
        else:
            self._train_status.setText("❌ Training failed — see log above.")
        self._refresh_model_status()
        self._refresh_buffer_stats()

    def _on_reload_models(self) -> None:
        """Emit signal so the main window can reload controller references."""
        self._refresh_model_status()
        self.models_updated.emit()
        self._train_status.setText("Models reloaded — next inference run will use updated weights.")
