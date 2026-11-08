"""
audio_window.py

The "Audio playback" window (View menu of the observer). It is a free
floating window: move it anywhere, even to another monitor. Closing it only
hides it; the audio keeps playing and the settings are kept until the
program ends.

It is just a view on the shared AudioPlayer: every control writes straight
into the player, and a timer reads the meters back.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from audio_player import AudioPlayer
from scanner_ui import (
    BORDER_COLOR,
    DISABLED_TEXT_COLOR,
    FFT_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    WINDOW_BACKGROUND_COLOR,
)


METER_INTERVAL_MS = 100
MIN_PREROLL_S = 0.10        # shorter than one lump of audio from the worker (~55 ms) cannot play smoothly
MAX_PREROLL_S = 2.00
LEVEL_WARNING_COLOR = "#FFAA00"


class AudioWindow(QWidget):
    def __init__(self, player: AudioPlayer, parent=None):
        super().__init__(parent, Qt.Window)

        self.player = player

        self.setWindowTitle("Audio playback")
        self.resize(380, 360)

        self._build_ui()
        self._apply_theme()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(METER_INTERVAL_MS)
        self._refresh()

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- controls ---------------------------------------------------
        controls = QGroupBox("Output")
        form = QFormLayout(controls)

        self.device_combo = QComboBox()
        self.device_combo.addItem("System default", None)

        for index, name in AudioPlayer.devices():
            self.device_combo.addItem(name, index)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(int(round(self.player.volume * 100)))
        self.volume_label = QLabel(f"{self.volume_slider.value()}%")

        volume_widget = QWidget()
        volume_layout = QHBoxLayout(volume_widget)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)

        self.mute_checkbox = QCheckBox("Mute")
        self.mute_checkbox.setChecked(self.player.muted)

        self.preroll_spin = QDoubleSpinBox()
        self.preroll_spin.setRange(MIN_PREROLL_S, MAX_PREROLL_S)
        self.preroll_spin.setSingleStep(0.05)
        self.preroll_spin.setDecimals(2)
        self.preroll_spin.setSuffix(" s")
        self.preroll_spin.setValue(self.player.buffer.preroll_s)
        self.preroll_spin.setToolTip(
            "Audio queued before playback starts. Longer = smoother, but more delay."
        )

        form.addRow("Device:", self.device_combo)
        form.addRow("Volume:", volume_widget)
        form.addRow("", self.mute_checkbox)
        form.addRow("Pre-roll:", self.preroll_spin)

        layout.addWidget(controls)

        # --- meters -----------------------------------------------------
        meters = QGroupBox("Status")
        meters_form = QFormLayout(meters)

        self.buffer_bar = QProgressBar()
        self.buffer_bar.setRange(0, int(self.player.buffer.max_s * 1000))
        self.buffer_bar.setFormat("%v ms")

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setFormat("")

        self.state_label = QLabel("")
        self.counters_label = QLabel("")
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)

        self.reset_button = QPushButton("Reset counters")

        meters_form.addRow("Buffer:", self.buffer_bar)
        meters_form.addRow("Level:", self.level_bar)
        meters_form.addRow("State:", self.state_label)
        meters_form.addRow("Problems:", self.counters_label)
        meters_form.addRow("", self.error_label)
        meters_form.addRow("", self.reset_button)

        layout.addWidget(meters)
        layout.addStretch()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {WINDOW_BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}

            QGroupBox {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                margin-top: 10px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}

            QPushButton, QComboBox, QDoubleSpinBox {{
                background-color: {PANEL_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 2px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {PANEL_COLOR};
                selection-background-color: {BORDER_COLOR};
            }}

            QProgressBar {{
                background-color: #000000;
                border: 1px solid {BORDER_COLOR};
                text-align: center;
            }}

            QProgressBar::chunk {{
                background-color: {FFT_COLOR};
            }}

            QPushButton:disabled, QLabel:disabled {{
                color: {DISABLED_TEXT_COLOR};
            }}
        """)

    # ==================================================================
    # Wiring
    # ==================================================================

    def _connect_signals(self):
        self.volume_slider.valueChanged.connect(self._on_volume)
        self.mute_checkbox.toggled.connect(self.player.set_muted)
        self.preroll_spin.valueChanged.connect(self.player.set_preroll)
        self.device_combo.currentIndexChanged.connect(self._on_device)
        self.reset_button.clicked.connect(self._reset_counters)

    def _on_volume(self, value):
        self.volume_label.setText(f"{value}%")
        self.player.set_volume(value / 100.0)

    def _on_device(self, _index):
        self.player.set_device(self.device_combo.currentData())

    def _reset_counters(self):
        self.player.buffer.underruns = 0
        self.player.buffer.overruns = 0
        self._refresh()

    # ==================================================================
    # Meters
    # ==================================================================

    def _refresh(self):
        buffer = self.player.buffer

        self.buffer_bar.setValue(int(buffer.buffered_s * 1000))

        level = min(1.0, self.player.level)
        self.level_bar.setValue(int(level * 100))

        state = buffer.state
        if self.player.muted:
            state += " (muted)"
        self.state_label.setText(state)

        self.counters_label.setText(
            f"underruns {buffer.underruns}    overruns {buffer.overruns}"
        )

        self.error_label.setText(
            f"Audio unavailable: {self.player.error}" if self.player.error else ""
        )
