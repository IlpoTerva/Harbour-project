"""
Jetson-side standalone Qt GUI for the Harbour Agent.

Runs the full pipeline (vision, STT, TTS, LLM, SQLite) directly on the
Jetson without any HTTP layer.  Requires a USB microphone and speaker.

Run (from the project root):
    python UI/jetson_frontend.py [--config utils/config.yaml] [--language en]
"""

import argparse
import logging
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional

import numpy as np

# PySide6 must be imported (and QApplication created) BEFORE cv2 is imported.
# OpenCV on Jetson/Ubuntu links against Qt5; importing it before Qt6 is
# initialised loads two incompatible Qt runtimes and causes heap corruption
# ("free(): invalid pointer").  cv2 is therefore imported lazily, inside the
# two methods that need it, after QApplication already exists.
from PySide6.QtCore import Qt, QObject, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHeaderView,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# LocalOrchestrator (and its transitive cv2 / VisionPipeline imports) must
# also be imported AFTER QApplication exists — see main() below.
from i18n import SUPPORTED_LANGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Signal containers ─────────────────────────────────────────────────────────

class _WorkerSignals(QObject):
    vision_ready   = Signal(object)        # Optional[Dict] — emitted after plate detection
    log_message    = Signal(str)
    status_update  = Signal(str, str)      # (text, colour_hex)
    button_enable  = Signal(bool)
    flow_complete  = Signal(object, object)  # (result_dict, vision_output)
    error_occurred = Signal(str)


class _DbSignals(QObject):
    data_ready = Signal(list)
    error      = Signal(str)


# ── Database dialog ───────────────────────────────────────────────────────────

class DatabaseDialog(QDialog):
    def __init__(self, orchestrator: Any, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Truck Database")
        self.resize(640, 360)
        self._orchestrator = orchestrator
        self._signals = _DbSignals()
        self._signals.data_ready.connect(self._populate)
        self._signals.error.connect(self._show_error)
        self._build_ui()
        self._fetch()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Registered Trucks")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._status_label = QLabel("Loading…")
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(
            ["Plate", "Driver Name", "Cargo", "Dock", "Arrival Window"]
        )
        self._tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setStretchLastSection(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._tree)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(
            "background-color: #27ae60; color: white; "
            "font-weight: bold; padding: 6px 20px;"
        )
        refresh_btn.clicked.connect(self._fetch)
        layout.addWidget(refresh_btn, alignment=Qt.AlignCenter)

    def _fetch(self) -> None:
        self._status_label.setText("Loading…")

        def _worker():
            try:
                rows = self._orchestrator.list_all_plates()
                self._signals.data_ready.emit(rows)
            except Exception as exc:
                self._signals.error.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(list)
    def _populate(self, rows: list) -> None:
        self._tree.clear()
        for row in rows:
            QTreeWidgetItem(self._tree, [
                row["plate"],
                row["driver_name"],
                row["cargo"],
                row["dock"],
                row["arrival_window"],
            ])
        self._status_label.setText(f"{len(rows)} record(s) loaded.")

    @Slot(str)
    def _show_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg}")


# ── Main window ───────────────────────────────────────────────────────────────

class JetsonGUI(QMainWindow):
    DISPLAY_SIZE = (400, 300)

    def __init__(self, orchestrator: Any) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.orchestrator.on_vision_result = self._on_vision_update_from_thread

        self._signals = _WorkerSignals()
        self._connect_signals()

        self.setWindowTitle("Harbor Gate: AI Logistics System (Local)")
        self._build_ui()

    def _connect_signals(self) -> None:
        self._signals.vision_ready.connect(self._slot_vision_ready)
        self._signals.log_message.connect(self._slot_log)
        self._signals.status_update.connect(self._slot_set_status)
        self._signals.button_enable.connect(self._slot_set_button_enabled)
        self._signals.flow_complete.connect(self._slot_on_complete)
        self._signals.error_occurred.connect(self._slot_on_error)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ── Left column: image + status + buttons ─────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        self._image_label = QLabel()
        self._image_label.setFixedSize(*self.DISPLAY_SIZE)
        self._image_label.setStyleSheet("background-color: #2c3e50;")
        self._image_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self._image_label)

        self._status_label = QLabel("System Ready")
        self._status_label.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        self._status_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self._status_label)

        btn_style = (
            "background-color: #27ae60; color: white; "
            "font-size: 12px; font-weight: bold; padding: 10px;"
        )

        self._import_btn = QPushButton("IMPORT TRUCK IMAGE")
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.setMinimumHeight(44)
        self._import_btn.clicked.connect(self._import_image)
        left.addWidget(self._import_btn)

        self._db_btn = QPushButton("VIEW DATABASE")
        self._db_btn.setStyleSheet(btn_style)
        self._db_btn.setMinimumHeight(44)
        self._db_btn.clicked.connect(self._open_db_window)
        left.addWidget(self._db_btn)

        left.addStretch()
        main_layout.addLayout(left)

        # ── Right column: log ─────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        log_title = QLabel("Gate Assistant Logs")
        log_title.setFont(QFont("Helvetica", 12))
        log_title.setAlignment(Qt.AlignCenter)
        right.addWidget(log_title)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFont(QFont("Courier New", 10))
        self._log_box.setMinimumWidth(300)
        right.addWidget(self._log_box)

        main_layout.addLayout(right)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object)
    def _slot_vision_ready(self, vision_output: Optional[Dict]) -> None:
        if vision_output is not None:
            self._display_image(vision_output["visual"])

    @Slot(str)
    def _slot_log(self, message: str) -> None:
        self._log_box.append(f"> {message}")
        self._log_box.moveCursor(QTextCursor.MoveOperation.End)

    @Slot(str, str)
    def _slot_set_status(self, text: str, colour: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {colour}; font-size: 16px; font-weight: bold;"
        )

    @Slot(bool)
    def _slot_set_button_enabled(self, enabled: bool) -> None:
        self._import_btn.setEnabled(enabled)

    @Slot(object, object)
    def _slot_on_complete(
        self, result: Dict[str, Any], vision_output: Optional[Dict[str, Any]]
    ) -> None:
        self._import_btn.setEnabled(True)

        if vision_output is not None:
            self._display_image(vision_output["visual"])

        status = result["status"]
        if status == "success":
            db = result["db_entry"]
            self._slot_set_status(f"ENTRY PERMITTED: {db['plate']}", "green")
            self._slot_log(
                f"Verification complete.\n"
                f"   Driver: {db['driver_name']}\n"
                f"   Cargo:  {db['cargo']}\n"
                f"   Dock:   {db['dock']}\n"
                f"   Window: {db['arrival_window']}"
            )
        elif status == "alert_worker":
            plate = result.get("plate", "UNKNOWN")
            self._slot_set_status("ALERT: UNREGISTERED VEHICLE", "red")
            self._slot_log(
                f"Plate '{plate}' is not in the database.\n"
                "   Manual intervention required."
            )
        elif status == "name_mismatch":
            plate = result.get("plate", "UNKNOWN")
            self._slot_set_status("ALERT: NAME MISMATCH", "red")
            self._slot_log(
                f"Driver name verification failed for '{plate}'.\n"
                "   Manual intervention required."
            )
        else:
            self._slot_set_status("UNKNOWN STATUS", "orange")
            self._slot_log(f"Unexpected status: {status!r}")

    @Slot(str)
    def _slot_on_error(self, err_msg: str) -> None:
        self._import_btn.setEnabled(True)
        self._slot_set_status("SYSTEM ERROR", "red")
        self._slot_log(f"Fatal error: {err_msg}")

    # ── Image display ─────────────────────────────────────────────────────────

    def _display_image(self, image: np.ndarray) -> None:
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR → RGB without cv2
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.DISPLAY_SIZE[0], self.DISPLAY_SIZE[1],
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_vision_update_from_thread(self, vision_output: Optional[Dict]) -> None:
        """Called from the worker thread; bounces the image update to the main thread."""
        self._signals.vision_ready.emit(vision_output)

    def _import_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Truck Image", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if not path:
            return
        import cv2  # noqa: PLC0415 — lazy: must run after QApplication exists
        raw_image = cv2.imread(path)
        if raw_image is None:
            self._slot_log(f"Failed to read image: {path}")
            return
        self._display_image(raw_image)
        self._import_btn.setEnabled(False)
        self._slot_set_status("PROCESSING…", "orange")
        self._slot_log("Entry flow started. Running vision and voice verification…")
        threading.Thread(target=self._flow_worker, args=(raw_image,), daemon=True).start()

    def _open_db_window(self) -> None:
        DatabaseDialog(self.orchestrator, parent=self).exec()

    def _flow_worker(self, raw_image: np.ndarray) -> None:
        try:
            result, vision_output = self.orchestrator.run_automated_entry(raw_image)
        except Exception as exc:
            logger.exception("LocalOrchestrator flow raised an exception.")
            self._signals.error_occurred.emit(str(exc))
            return
        self._signals.flow_complete.emit(result, vision_output)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Harbour Agent Jetson local frontend")
    parser.add_argument(
        "--config",
        default="utils/config.yaml",
        help="Path to config.yaml (default: utils/config.yaml)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=sorted(SUPPORTED_LANGS),
        help="Default TTS/STT language (default: en)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Allow Ctrl+C to reach Python: install handler + timer so Qt yields periodically
    signal.signal(signal.SIGINT, lambda *_: QApplication.quit())
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)

    splash = QLabel("Loading models, please wait…")
    splash.setAlignment(Qt.AlignCenter)
    splash.setWindowTitle("Harbour Agent")
    splash.resize(320, 90)
    splash.setStyleSheet("font-size: 14px; padding: 20px;")
    splash.show()
    app.processEvents()

    # Import cv2-dependent code HERE — after QApplication exists — so Qt6 is
    # fully initialised before OpenCV's Qt5 linkage is loaded into the process.
    from local_orchestrator import LocalOrchestrator  # noqa: PLC0415
    from scripts.helpers import read_config            # noqa: PLC0415

    config = read_config(args.config)
    orchestrator = LocalOrchestrator(config=config, default_language=args.language)

    # Show main window before closing splash so Qt never sees "no windows open"
    # (quitOnLastWindowClosed would otherwise queue a quit before app.exec() starts)
    window = JetsonGUI(orchestrator=orchestrator)
    window.show()
    splash.close()

    app.aboutToQuit.connect(orchestrator.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
