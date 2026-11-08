"""
main.py

One window for the whole application:

    [ Scanning | Observing ]            <- mode switch
    ---------------------------------
    the scanner UI  or  the observer UI

Only one mode owns the SDR at a time. The switch is disabled while either
mode is running (a scan, or a live observation), so the device is always
released before the other mode starts.

The scanner and the observer share ONE ChannelManager: the scanner adds
channels, the observer refines them and records their samples.
"""

import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from channel_manager import ChannelManager
from demod import ModulationRegistry
from observer_app import ObserverApplication
from observer_ui import ObserverUI
from scanner_app import ScannerApplication
from scanner_ui import (
    BORDER_COLOR,
    DISABLED_TEXT_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    WINDOW_BACKGROUND_COLOR,
    ScannerUI,
)

SCANNING = 0
OBSERVING = 1


class MainWindow(QMainWindow):
    def __init__(self, scanner_app, observer_app):
        super().__init__()

        self.scanner_app = scanner_app
        self.observer_app = observer_app

        self.setWindowTitle("SDR Spectrum Scanner / Observer")
        self.resize(1500, 900)

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {WINDOW_BACKGROUND_COLOR}; }}

            QPushButton {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 4px 16px;
            }}

            QPushButton:checked {{
                background-color: {BORDER_COLOR};
                border: 1px solid {TEXT_COLOR};
            }}

            QPushButton:disabled {{ color: {DISABLED_TEXT_COLOR}; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(4)

        # --- mode switch ------------------------------------------------
        self.scanning_button = QPushButton("Scanning")
        self.observing_button = QPushButton("Observing")

        for button in (self.scanning_button, self.observing_button):
            button.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.scanning_button, SCANNING)
        self.mode_group.addButton(self.observing_button, OBSERVING)
        self.scanning_button.setChecked(True)

        bar = QHBoxLayout()
        bar.addWidget(self.scanning_button)
        bar.addWidget(self.observing_button)
        bar.addStretch()
        layout.addLayout(bar)

        # --- the two UIs ------------------------------------------------
        self.stack = QStackedWidget()
        self.stack.addWidget(scanner_app.ui)
        self.stack.addWidget(observer_app.ui)
        layout.addWidget(self.stack, stretch=1)

        self.mode_group.idClicked.connect(self._mode_requested)

        # Mode buttons are only usable while nothing holds the SDR.
        self._busy_timer = QTimer(self)
        self._busy_timer.timeout.connect(self._update_busy)
        self._busy_timer.start(150)

    def _mode_requested(self, mode):
        if self._busy():
            self._show_current_mode_button()
            return

        if mode == OBSERVING:
            self.observer_app.refresh_channels()     # pick up channels the scanner found

        self.stack.setCurrentIndex(mode)

    def _busy(self):
        return self.scanner_app.worker is not None or self.observer_app.is_running()

    def _update_busy(self):
        busy = self._busy()

        self.scanning_button.setEnabled(not busy)
        self.observing_button.setEnabled(not busy)

    def _show_current_mode_button(self):
        button = self.mode_group.button(self.stack.currentIndex())
        if button is not None:
            button.setChecked(True)

    def shutdown(self):
        self.scanner_app.close()
        self.observer_app.close()


def main():
    app = QApplication(sys.argv)

    manager = ChannelManager("channels")
    registry = ModulationRegistry("modulations")

    scanner_app = ScannerApplication(ui=ScannerUI(), channels=manager, show=False)
    observer_app = ObserverApplication(ObserverUI(), manager, registry)

    window = MainWindow(scanner_app, observer_app)
    window.show()

    app.aboutToQuit.connect(window.shutdown)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()