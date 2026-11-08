from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg

from known_ranges import ranges_manager as rf_range
# from ranges_manager import rf_range

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ======================================================================
# Colours and display constants (edit here)
# ======================================================================

WINDOW_BACKGROUND_COLOR = "#1E1E1E" # whole window
PANEL_COLOR = "#2B2B2B"             # buttons, inputs, lists, menus
BORDER_COLOR = "#555555"            # group boxes and input borders
DISABLED_TEXT_COLOR = "#777777"     # text of disabled controls

BACKGROUND_COLOR = "#000000"        # FFT, range strip and waterfall background
TEXT_COLOR = "#C8C8C8"              # all text: window, axis labels, tick numbers
AXIS_COLOR = "#C8C8C8"              # axis lines (the grid follows this too)

FFT_COLOR = "#00FFFF"               # live FFT curve (cyan)
MAX_HOLD_COLOR = "#FFFF00"          # max hold curve (yellow)
CENTER_MARKER_COLOR = "#FFFFFF"     # centre-frequency marker (white)
BEACON_COLOR = "#00FF00"            # beacon markers (green)
BEACON_BAND_ALPHA = 40               # beacon bandwidth fill opacity, 0 to 255
THRESHOLD_COLOR = "#FF0000"         # threshold curve (red)

GRID_ALPHA = 0.2                    # FFT grid opacity, 0.0 to 1.0
RANGE_FILL_ALPHA = 60               # frequency-range fill opacity, 0 to 255

# ----------------------------------------------------------------------
# Waterfall colour ramp
#
# The colour ramp is calibrated around the threshold:
#
#   lowest colour  = threshold - WATERFALL_DB_BELOW_THRESHOLD
#   highest colour = threshold + WATERFALL_DB_ABOVE_THRESHOLD
#
# The threshold used here is a single number: the mean of the
# threshold curve (see set_threshold).
# ----------------------------------------------------------------------

WATERFALL_DB_BELOW_THRESHOLD = 30.0   # dB shown below the threshold
WATERFALL_DB_ABOVE_THRESHOLD = 30.0   # dB shown above the threshold

# Colour stops: (position 0..1, (R, G, B)).
# Position 0.5 corresponds to the threshold when the two dB spans
# above are equal.
WATERFALL_COLOR_STOPS = [
    (0.00, (0, 0, 0)),          # black (well below threshold)
    (0.30, (0, 0, 10)),         # dark blue
    (0.50, (0, 0, 255)),        # blue    <- threshold
    (0.65, (0, 255, 255)),      # cyan
    (0.80, (255, 255, 0)),      # yellow
    (0.90, (255, 0, 0)),        # red
    (1.00, (255, 255, 255)),    # white (strongest)
]

# Value used for waterfall rows that have no data yet. It is far below
# any realistic level, so it always maps to the first (black) colour.
WATERFALL_EMPTY_VALUE = -1000.0


class ScannerUI(QMainWindow):
    # ------------------------------------------------------------------
    # Signals: UI -> main.py
    # ------------------------------------------------------------------

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    single_shot_requested = pyqtSignal()

    start_frequency_changed = pyqtSignal(float, str)
    stop_frequency_changed = pyqtSignal(float, str)

    gain_changed = pyqtSignal(int)

    sample_rate_changed = pyqtSignal(float, str)
    smoothness_changed = pyqtSignal(int)
    threshold_changed = pyqtSignal(float)
    dynamic_threshold_changed = pyqtSignal(bool)

    beacon_bandwidth_changed = pyqtSignal(float, str)

    # Emitted with the full file path chosen in the Save dialog.
    # main.py writes the beacon list to that path
    # (use os.path.dirname(path) if only the directory is needed).
    export_beacons_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Wideband SDR Scanner")
        self.resize(1400, 800)

        # --------------------------------------------------------------
        # Internal display state
        # --------------------------------------------------------------

        self._waterfall_rows = 300
        self._waterfall_buffer = None
        self._waterfall_frequency_range = None

        # Threshold used to calibrate the waterfall colour ramp
        # (a single number: the mean of the threshold curve).
        self._threshold_db = -60.0

        self._max_hold_values = None

        self._beacons = []              # list of beacon dicts
        self._beacon_frequencies = []   # frequency of each beacon, in Hz
        self.beacon_markers = []
        self.beacon_band_items = []

        # Frequency ranges
        self._frequency_ranges = []
        self._frequency_range_items = []

        self._build_ui()
        self._build_menus()
        self._connect_internal_signals()
        self._connect_hover_signals()

        # Initial state
        self._show_blank_waterfall()
        self.set_acquisition_active(False)

    # ==================================================================
    # PUBLIC UI API
    # ==================================================================

    def set_center_frequency(self, frequency_hz: float):
        """
        Update the displayed centre frequency and its FFT marker.

        This does NOT change the FFT frequency span/window.
        """

        frequency_hz = float(frequency_hz)

        self.center_frequency_label.setText(self._format_frequency(frequency_hz))
        self.center_frequency_marker.setValue(frequency_hz)

    def set_frame(
        self,
        powers: Sequence[float],
        frequencies_hz: Sequence[float],
        max_hold: Sequence[float] | None = None,
        threshold_curve: Sequence[float] | None = None,
    ):
        """
        Display one new FFT frame.

        powers and frequencies_hz must have the same length.

        max_hold is accepted for compatibility with the worker signal but
        is not used: the UI keeps its own max hold trace.

        threshold_curve (optional) is the per-bin threshold, same length
        as powers. It is drawn as a curve on the FFT.

        Every call:
            - updates the FFT, the max hold trace and the threshold curve
            - advances the waterfall by exactly one row
            - updates beacon and frequency-range visibility
        """

        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        powers = np.asarray(powers, dtype=np.float32)

        if frequencies.size == 0:
            return

        if frequencies.size != powers.size:
            raise ValueError("powers and frequencies_hz must have the same length")

        # --------------------------------------------------------------
        # FFT
        # --------------------------------------------------------------

        self.fft_curve.setData(frequencies, powers)

        # --------------------------------------------------------------
        # Max hold
        # --------------------------------------------------------------

        if (
            self._max_hold_values is None
            or self._max_hold_values.shape != powers.shape
        ):
            self._max_hold_values = powers.copy()
        else:
            np.maximum(
                self._max_hold_values,
                powers,
                out=self._max_hold_values,
            )

        self.max_hold_curve.setData(frequencies, self._max_hold_values)

        # --------------------------------------------------------------
        # Threshold curve
        # --------------------------------------------------------------

        if threshold_curve is not None:
            threshold = np.asarray(threshold_curve, dtype=np.float32)

            if threshold.size == frequencies.size:
                self.threshold_curve.setData(frequencies, threshold)

        # The incoming frame defines the currently visible
        # frequency window.
        x_min = float(np.min(frequencies))
        x_max = float(np.max(frequencies))

        if x_max > x_min:
            self.fft_plot.setXRange(x_min, x_max, padding=0.0)

        # --------------------------------------------------------------
        # Waterfall
        # --------------------------------------------------------------

        self._update_waterfall(powers, x_min, x_max)

        # --------------------------------------------------------------
        # Beacon visibility
        # --------------------------------------------------------------

        self._update_beacon_marker_visibility(x_min, x_max)

        # --------------------------------------------------------------
        # Frequency-range visibility
        # --------------------------------------------------------------

        self._update_frequency_range_visibility(x_min, x_max)

    def set_beacons(self, beacons: Sequence[dict]):
        """
        Replace the current beacon list and FFT beacon markers.

        Each beacon is a dict:

            {
                "frequency":   centre frequency in Hz,
                "bandwidth":   bandwidth in Hz,
                "start_index": first FFT bin of the beacon,
                "end_index":   last FFT bin of the beacon,
            }

        "frequency" is required. The other keys are optional for
        display purposes and may be None.
        """

        self._beacons = [dict(beacon) for beacon in beacons]

        self._beacon_frequencies = [
            float(beacon["frequency"]) for beacon in self._beacons
        ]

        # Left-side list: "frequency  (BW bandwidth)"
        self.beacon_list.clear()

        for beacon in self._beacons:

            text = self._format_frequency(float(beacon["frequency"]))

            bandwidth = beacon.get("bandwidth")

            if bandwidth is not None:
                text += f"  (BW {self._format_frequency(float(bandwidth))})"

            item = QListWidgetItem(text)

            # Keep the full dict on the item for later use.
            item.setData(Qt.UserRole, beacon)

            self.beacon_list.addItem(item)

        # Remove old markers and bandwidth regions.
        for marker in self.beacon_markers:
            self.fft_plot.removeItem(marker)

        self.beacon_markers.clear()
        for band in self.beacon_band_items:
            if band is not None:
                self.fft_plot.removeItem(band)

        self.beacon_band_items.clear()

        band_color = QColor(BEACON_COLOR)
        band_color.setAlpha(BEACON_BAND_ALPHA)

        # Draw translucent bandwidth regions behind the center markers.
        for beacon in self._beacons:
            frequency = float(beacon["frequency"])
            bandwidth = beacon.get("bandwidth")

            if bandwidth is None or float(bandwidth) <= 0:
                self.beacon_band_items.append(None)
                continue

            half_bandwidth = float(bandwidth) / 2.0
            band = pg.LinearRegionItem(
                values=(frequency - half_bandwidth, frequency + half_bandwidth),
                orientation="vertical",
                movable=False,
                brush=QBrush(band_color),
                pen=pg.mkPen(BEACON_COLOR, width=0.5),
            )
            band.setZValue(-10)
            self.fft_plot.addItem(band)
            self.beacon_band_items.append(band)

        # Create new markers (green, dotted)
        for frequency in self._beacon_frequencies:

            marker = pg.InfiniteLine(
                pos=frequency,
                angle=90,
                movable=False,
                pen=pg.mkPen(BEACON_COLOR, width=1, style=Qt.DotLine),
            )

            self.fft_plot.addItem(marker)
            self.beacon_markers.append(marker)

        # Apply visibility based on current FFT window.
        if self._waterfall_frequency_range is not None:
            x_min, x_max = self._waterfall_frequency_range
            self._update_beacon_marker_visibility(x_min, x_max)

    def set_frequency_ranges(self, ranges: Sequence[rf_range]):
        """
        Replace the currently displayed frequency ranges.

        Each RF range contains:

            start_freq   Start frequency in Hz.
            end_freq     End frequency in Hz.
            info         Text displayed inside the range.
            hex_color    Colour used for the range border, fill and text.

        The ranges are displayed in a dedicated strip between the FFT
        and the waterfall. The strip shares the FFT X axis.
        """

        self._frequency_ranges = list(ranges)

        # Remove previous range graphics
        for fill, text in self._frequency_range_items:
            self.frequency_ranges_plot.removeItem(fill)
            self.frequency_ranges_plot.removeItem(text)

        self._frequency_range_items.clear()

        # Create new range graphics
        for frequency_range in self._frequency_ranges:

            start = float(frequency_range.start_freq)
            end = float(frequency_range.end_freq)

            if end <= start:
                continue

            color = QColor(frequency_range.hex_color)

            fill_color = QColor(color)
            fill_color.setAlpha(RANGE_FILL_ALPHA)

            fill = pg.LinearRegionItem(
                values=(start, end),
                orientation="vertical",
                movable=False,
                brush=QBrush(fill_color),
                pen=QPen(color, 1.5),
            )

            self.frequency_ranges_plot.addItem(fill)

            text = pg.TextItem(
                text=frequency_range.info,
                color=color,
                anchor=(0.5, 0.5),
            )

            text.setPos((start + end) / 2.0, 0.5)

            self.frequency_ranges_plot.addItem(text)

            # Store both items as a pair.
            self._frequency_range_items.append((fill, text))

        self.frequency_ranges_plot.setYRange(0, 1, padding=0.0)

        # Apply current FFT visibility
        if self._waterfall_frequency_range is not None:
            x_min, x_max = self._waterfall_frequency_range
            self._update_frequency_range_visibility(x_min, x_max)

    def set_acquisition_active(self, active: bool):
        """
        Enable/disable the Frequency and Settings sections.

        Controls remain enabled so STOP can always be pressed.
        """

        active = bool(active)

        self.frequency_group.setEnabled(not active)
        self.settings_group.setEnabled(not active)
        self.controls_group.setEnabled(True)

        # Keep STOP available while acquisition is active.
        self.stop_button.setEnabled(active)

        # Start / Single Shot aren't useful while acquisition is active.
        self.start_button.setEnabled(not active)
        self.single_shot_button.setEnabled(not active)

    def set_configuration(
        self,
        *,
        start_frequency: float,
        stop_frequency: float,
        gain: int,
        sample_rate: float,
        smoothness: int,
        threshold: float,
        dynamic_threshold: bool,
        beacon_bandwidth: float,
    ):
        """
        Set all configuration controls from values expressed in Hz.
        """

        controls = (
            self.start_frequency_spin,
            self.start_frequency_unit,
            self.stop_frequency_spin,
            self.stop_frequency_unit,
            self.gain_slider,
            self.sample_rate_spin,
            self.sample_rate_unit,
            self.smoothness_spin,
            self.threshold_spin,
            self.dynamic_threshold_checkbox,
            self.beacon_bandwidth_spin,
            self.beacon_bandwidth_unit,
        )

        previous_blocked = [control.blockSignals(True) for control in controls]

        try:
            self._set_frequency_control(
                self.start_frequency_spin,
                self.start_frequency_unit,
                start_frequency,
            )

            self._set_frequency_control(
                self.stop_frequency_spin,
                self.stop_frequency_unit,
                stop_frequency,
            )

            self.gain_slider.setValue(int(gain))

            self._set_frequency_control(
                self.sample_rate_spin,
                self.sample_rate_unit,
                sample_rate,
            )

            self.smoothness_spin.setValue(int(smoothness))

            self.threshold_spin.setValue(float(threshold))
            # Signals are blocked, so update the waterfall colour
            # calibration explicitly.
            self._set_threshold_state(float(threshold))

            self.dynamic_threshold_checkbox.setChecked(bool(dynamic_threshold))

            self._set_frequency_control(
                self.beacon_bandwidth_spin,
                self.beacon_bandwidth_unit,
                beacon_bandwidth,
            )

        finally:
            for control, blocked in zip(controls, previous_blocked):
                control.blockSignals(blocked)

        self._on_gain_changed(self.gain_slider.value())

        self._emit_start_frequency()
        self._emit_stop_frequency()
        self._emit_sample_rate()

        self.smoothness_changed.emit(self.smoothness_spin.value())
        self.threshold_changed.emit(self.threshold_spin.value())
        self.dynamic_threshold_changed.emit(
            self.dynamic_threshold_checkbox.isChecked()
        )

        self._emit_beacon_bandwidth()

    def set_threshold(self, threshold_db: float):
        """
        Set the single threshold number (in dB) used to calibrate the
        waterfall colour ramp. main.py sends the mean of the threshold
        curve here after every sweep.

        The threshold CURVE itself is drawn in set_frame().
        """

        self._set_threshold_state(float(threshold_db))

    @staticmethod
    def _set_frequency_control(value_widget, unit_widget, frequency_hz):

        frequency_hz = float(frequency_hz)

        if abs(frequency_hz) >= 1e9:
            unit = "GHz"
        elif abs(frequency_hz) >= 1e6:
            unit = "MHz"
        elif abs(frequency_hz) >= 1e3:
            unit = "kHz"
        else:
            unit = "Hz"

        multipliers = {
            "Hz": 1.0,
            "kHz": 1e3,
            "MHz": 1e6,
            "GHz": 1e9,
        }

        unit_widget.setCurrentText(unit)
        value_widget.setValue(frequency_hz / multipliers[unit])

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):

        self._apply_theme()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --------------------------------------------------------------
        # LEFT COLUMN - Beacons
        # --------------------------------------------------------------

        beacon_group = QGroupBox("Beacons")
        beacon_layout = QVBoxLayout(beacon_group)

        self.beacon_list = QListWidget()
        self.beacon_list.setSelectionMode(QListWidget.NoSelection)

        beacon_layout.addWidget(self.beacon_list)
        main_layout.addWidget(beacon_group, stretch=1)

        # --------------------------------------------------------------
        # CENTRE COLUMN - FFT + Ranges + Waterfall + hover read-out
        # --------------------------------------------------------------

        display_widget = QWidget()

        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)

        # ==============================================================
        # FFT
        # ==============================================================

        self.fft_plot = pg.PlotWidget()
        self._style_plot(self.fft_plot)

        self.fft_plot.setLabel("left", "Power", units="dB", color=TEXT_COLOR)
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz", color=TEXT_COLOR)
        self.fft_plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)

        # Live FFT curve
        self.fft_curve = self.fft_plot.plot(
            pen=pg.mkPen(FFT_COLOR, width=1)
        )

        # Max hold curve, drawn on top
        self.max_hold_curve = self.fft_plot.plot(
            pen=pg.mkPen(MAX_HOLD_COLOR, width=1)
        )

        # Threshold curve (dashed, one value per frequency bin, in dB)
        self.threshold_curve = self.fft_plot.plot(
            pen=pg.mkPen(THRESHOLD_COLOR, width=1, style=Qt.DashLine)
        )

        # Centre-frequency marker (dashed)
        self.center_frequency_marker = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen(CENTER_MARKER_COLOR, width=1, style=Qt.DashLine),
        )

        self.fft_plot.addItem(self.center_frequency_marker)

        display_layout.addWidget(self.fft_plot, stretch=2)

        # ==============================================================
        # Frequency Ranges
        # ==============================================================

        self.frequency_ranges_plot = pg.PlotWidget()
        self._style_plot(self.frequency_ranges_plot)

        self.frequency_ranges_plot.setFixedHeight(22)
        self.frequency_ranges_plot.setContentsMargins(0, 0, 0, 0)
        self.frequency_ranges_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)

        # Same X axis as FFT.
        self.frequency_ranges_plot.setXLink(self.fft_plot)

        self.frequency_ranges_plot.hideAxis("left")
        self.frequency_ranges_plot.hideAxis("bottom")

        # No interaction in the range strip.
        self.frequency_ranges_plot.setMouseEnabled(x=False, y=False)
        self.frequency_ranges_plot.setMenuEnabled(False)

        display_layout.addWidget(self.frequency_ranges_plot, stretch=0)

        # ==============================================================
        # Waterfall
        # ==============================================================

        self.waterfall_plot = pg.PlotWidget()
        self._style_plot(self.waterfall_plot)

        self.waterfall_plot.invertY()
        self.waterfall_plot.setXLink(self.fft_plot)

        self.waterfall_plot.setLabel("left", "Time", color=TEXT_COLOR)
        self.waterfall_plot.setLabel(
            "bottom", "Frequency", units="Hz", color=TEXT_COLOR
        )
        self.waterfall_plot.hideAxis("left")

        self.waterfall_image = pg.ImageItem()

        # Colour ramp (black -> blue -> cyan -> yellow -> red -> white).
        colour_map = pg.ColorMap(
            [position for position, _ in WATERFALL_COLOR_STOPS],
            [colour for _, colour in WATERFALL_COLOR_STOPS],
        )
        self.waterfall_image.setLookupTable(
            colour_map.getLookupTable(0.0, 1.0, 256)
        )

        self.waterfall_plot.addItem(self.waterfall_image)

        display_layout.addWidget(self.waterfall_plot, stretch=3)

        # ==============================================================
        # Hover read-out (below the plots)
        # ==============================================================

        self.hover_label = QLabel(self._hover_text(None, None))
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        display_layout.addWidget(self.hover_label, stretch=0)

        main_layout.addWidget(display_widget, stretch=6)

        # --------------------------------------------------------------
        # RIGHT COLUMN - Controls
        # --------------------------------------------------------------

        right_widget = QWidget()

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # ==============================================================
        # Controls
        # ==============================================================

        self.controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(self.controls_group)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.single_shot_button = QPushButton("Single Shot")

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.single_shot_button)

        right_layout.addWidget(self.controls_group)

        # ==============================================================
        # Current centre frequency
        # ==============================================================

        frequency_display_group = QGroupBox("Current Frequency")
        frequency_display_layout = QVBoxLayout(frequency_display_group)

        self.center_frequency_label = QLabel("0.000 MHz")
        self.center_frequency_label.setAlignment(Qt.AlignCenter)

        frequency_display_layout.addWidget(self.center_frequency_label)

        right_layout.addWidget(frequency_display_group)

        # ==============================================================
        # Frequency
        # ==============================================================

        self.frequency_group = QGroupBox("Frequency")
        frequency_layout = QFormLayout(self.frequency_group)

        self.start_frequency_spin = self._make_frequency_spinbox()
        self.start_frequency_unit = self._make_unit_combo()

        self.stop_frequency_spin = self._make_frequency_spinbox()
        self.stop_frequency_unit = self._make_unit_combo()

        frequency_layout.addRow(
            "Start:",
            self._make_value_unit_widget(
                self.start_frequency_spin,
                self.start_frequency_unit,
            ),
        )

        frequency_layout.addRow(
            "Stop:",
            self._make_value_unit_widget(
                self.stop_frequency_spin,
                self.stop_frequency_unit,
            ),
        )

        right_layout.addWidget(self.frequency_group)

        # ==============================================================
        # Settings
        # ==============================================================

        self.settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(self.settings_group)

        # Gain ---------------------------------------------------------

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(50)

        self.gain_value_label = QLabel("50%")

        gain_widget = QWidget()
        gain_layout = QHBoxLayout(gain_widget)
        gain_layout.setContentsMargins(0, 0, 0, 0)

        gain_layout.addWidget(self.gain_slider)
        gain_layout.addWidget(self.gain_value_label)

        settings_layout.addRow("Gain:", gain_widget)

        # Sample rate --------------------------------------------------

        self.sample_rate_spin = self._make_frequency_spinbox()
        self.sample_rate_unit = self._make_unit_combo()

        settings_layout.addRow(
            "Sample Rate:",
            self._make_value_unit_widget(
                self.sample_rate_spin,
                self.sample_rate_unit,
            ),
        )

        # Smoothness ---------------------------------------------------

        self.smoothness_spin = QSpinBox()
        self.smoothness_spin.setRange(0, 100)
        self.smoothness_spin.setValue(0)

        settings_layout.addRow("Smoothness:", self.smoothness_spin)

        # Threshold (offset added to the local mean) -------------------

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-200.0, 100.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setSuffix(" dB")
        self.threshold_spin.setValue(self._threshold_db)

        settings_layout.addRow("Threshold offset:", self.threshold_spin)

        # Dynamic threshold -------------------------------------------

        self.dynamic_threshold_checkbox = QCheckBox("Dynamic Threshold")

        settings_layout.addRow("", self.dynamic_threshold_checkbox)

        # Beacon bandwidth ---------------------------------------------

        self.beacon_bandwidth_spin = self._make_frequency_spinbox()
        self.beacon_bandwidth_unit = self._make_unit_combo()

        settings_layout.addRow(
            "Beacon BW:",
            self._make_value_unit_widget(
                self.beacon_bandwidth_spin,
                self.beacon_bandwidth_unit,
            ),
        )

        right_layout.addWidget(self.settings_group)
        right_layout.addStretch()

        main_layout.addWidget(right_widget, stretch=2)

    def _apply_theme(self):
        """
        Style the whole window (background, text, group boxes, inputs,
        menus) from the colour constants at the top of the file.
        """

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {WINDOW_BACKGROUND_COLOR};
            }}

            QMenuBar, QMenu {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
            }}

            QMenuBar::item:selected, QMenu::item:selected {{
                background-color: {BORDER_COLOR};
            }}

            QLabel, QCheckBox {{
                color: {TEXT_COLOR};
            }}

            QGroupBox {{
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                margin-top: 10px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}

            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 2px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                selection-background-color: {BORDER_COLOR};
            }}

            QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled,
            QDoubleSpinBox:disabled, QLabel:disabled, QCheckBox:disabled,
            QGroupBox:disabled {{
                color: {DISABLED_TEXT_COLOR};
            }}
        """)

    @staticmethod
    def _style_plot(plot):
        """
        Apply the background, axis and text colours to a PlotWidget.
        """

        plot.setBackground(BACKGROUND_COLOR)

        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(AXIS_COLOR))
            axis.setTextPen(pg.mkPen(TEXT_COLOR))

    # ==================================================================
    # Menus
    # ==================================================================

    def _build_menus(self):

        menu_bar = self.menuBar()

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = menu_bar.addMenu("File")

        export_beacons_action = QAction("Export Beacons...", self)
        edit_ranges_action = QAction("Edit Ranges...", self)

        export_beacons_action.triggered.connect(self._export_beacons)

        # Placeholder for now.
        edit_ranges_action.triggered.connect(self._edit_ranges_placeholder)

        file_menu.addAction(export_beacons_action)
        file_menu.addAction(edit_ranges_action)

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = menu_bar.addMenu("View")

        reset_fft_action = QAction("Reset FFT View", self)
        clear_waterfall_action = QAction("Clear Waterfall", self)
        clear_max_hold_action = QAction("Clear Max Hold", self)

        reset_fft_action.triggered.connect(self._reset_fft_view)
        clear_waterfall_action.triggered.connect(self._clear_waterfall)
        clear_max_hold_action.triggered.connect(self._clear_max_hold)

        view_menu.addAction(reset_fft_action)
        view_menu.addAction(clear_waterfall_action)
        view_menu.addAction(clear_max_hold_action)

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _connect_internal_signals(self):

        # Controls
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.single_shot_button.clicked.connect(self.single_shot_requested.emit)

        # Frequency
        self.start_frequency_spin.valueChanged.connect(self._emit_start_frequency)
        self.start_frequency_unit.currentTextChanged.connect(
            self._emit_start_frequency
        )

        self.stop_frequency_spin.valueChanged.connect(self._emit_stop_frequency)
        self.stop_frequency_unit.currentTextChanged.connect(
            self._emit_stop_frequency
        )

        # Settings
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        self.sample_rate_spin.valueChanged.connect(self._emit_sample_rate)
        self.sample_rate_unit.currentTextChanged.connect(self._emit_sample_rate)

        self.smoothness_spin.valueChanged.connect(self.smoothness_changed.emit)

        self.threshold_spin.valueChanged.connect(self.threshold_changed.emit)
        # Keep the waterfall colour ramp in sync with manual edits
        # (until the next sweep delivers the real mean of the curve).
        self.threshold_spin.valueChanged.connect(self._set_threshold_state)

        self.dynamic_threshold_checkbox.toggled.connect(
            self.dynamic_threshold_changed.emit
        )

        self.beacon_bandwidth_spin.valueChanged.connect(
            self._emit_beacon_bandwidth
        )
        self.beacon_bandwidth_unit.currentTextChanged.connect(
            self._emit_beacon_bandwidth
        )

    def _connect_hover_signals(self):
        """
        Track the mouse over the FFT and the waterfall.

        SignalProxy limits the update rate so hovering never slows down
        the display. The proxies must be kept as attributes, otherwise
        they are garbage collected and stop working.
        """

        self._fft_hover_proxy = pg.SignalProxy(
            self.fft_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_fft_mouse_moved,
        )

        self._waterfall_hover_proxy = pg.SignalProxy(
            self.waterfall_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_waterfall_mouse_moved,
        )

    # ==================================================================
    # Input helpers
    # ==================================================================

    @staticmethod
    def _make_frequency_spinbox():

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000_000_000.0)
        spin.setDecimals(6)
        spin.setSingleStep(1.0)

        return spin

    @staticmethod
    def _make_unit_combo():

        combo = QComboBox()
        combo.addItems(["Hz", "kHz", "MHz", "GHz"])
        combo.setCurrentText("MHz")

        return combo

    @staticmethod
    def _make_value_unit_widget(value_widget, unit_widget):

        widget = QWidget()

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(value_widget, stretch=1)
        layout.addWidget(unit_widget)

        return widget

    # ==================================================================
    # Signal emitters
    # ==================================================================

    def _emit_start_frequency(self):

        self.start_frequency_changed.emit(
            self.start_frequency_spin.value(),
            self.start_frequency_unit.currentText(),
        )

    def _emit_stop_frequency(self):

        self.stop_frequency_changed.emit(
            self.stop_frequency_spin.value(),
            self.stop_frequency_unit.currentText(),
        )

    def _emit_sample_rate(self):

        self.sample_rate_changed.emit(
            self.sample_rate_spin.value(),
            self.sample_rate_unit.currentText(),
        )

    def _emit_beacon_bandwidth(self):

        self.beacon_bandwidth_changed.emit(
            self.beacon_bandwidth_spin.value(),
            self.beacon_bandwidth_unit.currentText(),
        )

    def _on_gain_changed(self, value):

        self.gain_value_label.setText(f"{value}%")
        self.gain_changed.emit(value)

    # ==================================================================
    # Hover read-out
    # ==================================================================

    def _on_fft_mouse_moved(self, event):
        """
        Mouse over the FFT: show the frequency and the power (dB)
        at the cursor position.
        """

        scene_pos = event[0]
        view_box = self.fft_plot.getPlotItem().vb

        if not view_box.sceneBoundingRect().contains(scene_pos):
            return

        point = view_box.mapSceneToView(scene_pos)

        self.hover_label.setText(self._hover_text(point.x(), point.y()))

    def _on_waterfall_mouse_moved(self, event):
        """
        Mouse over the waterfall: show the frequency under the cursor
        and the power (dB) stored in the waterfall at that point.
        """

        scene_pos = event[0]
        view_box = self.waterfall_plot.getPlotItem().vb

        if not view_box.sceneBoundingRect().contains(scene_pos):
            return

        point = view_box.mapSceneToView(scene_pos)

        self.hover_label.setText(
            self._hover_text(point.x(), self._waterfall_power_at(point.x(), point.y()))
        )

    def _waterfall_power_at(self, frequency_hz: float, row_position: float):
        """
        Return the stored power at a waterfall position, or None if there
        is no data there (outside the image or row not yet filled).
        """

        if self._waterfall_buffer is None or self._waterfall_frequency_range is None:
            return None

        x_min, x_max = self._waterfall_frequency_range

        if x_max <= x_min or not (x_min <= frequency_hz <= x_max):
            return None

        rows, columns = self._waterfall_buffer.shape

        row = int(row_position)

        if not (0 <= row < rows):
            return None

        column = int((frequency_hz - x_min) / (x_max - x_min) * columns)
        column = min(max(column, 0), columns - 1)

        value = float(self._waterfall_buffer[row, column])

        if value <= WATERFALL_EMPTY_VALUE / 2.0:
            return None

        return value

    def _hover_text(self, frequency_hz, power_db) -> str:

        frequency_text = (
            "--" if frequency_hz is None else self._format_frequency(frequency_hz)
        )
        power_text = "--" if power_db is None else f"{power_db:.1f} dB"

        return f"Frequency: {frequency_text}    Power: {power_text}"

    # ==================================================================
    # Waterfall
    # ==================================================================

    def _waterfall_levels(self):
        """
        Colour-ramp limits, calibrated around the threshold
        (the mean of the threshold curve).
        """

        return (
            self._threshold_db - WATERFALL_DB_BELOW_THRESHOLD,
            self._threshold_db + WATERFALL_DB_ABOVE_THRESHOLD,
        )

    def _set_threshold_state(self, threshold_db: float):
        """
        Store the threshold and recalibrate the waterfall colour ramp.
        """

        self._threshold_db = float(threshold_db)

        # Recolour the existing image immediately.
        self.waterfall_image.setLevels(self._waterfall_levels())

    def _draw_waterfall(self):
        """
        Push the waterfall buffer to the image, using the
        threshold-calibrated colour levels.
        """

        self.waterfall_image.setImage(
            self._waterfall_buffer,
            autoLevels=False,
            levels=self._waterfall_levels(),
            axisOrder="row-major",
        )

        if self._waterfall_frequency_range is not None:
            x_min, x_max = self._waterfall_frequency_range

            if x_max > x_min:
                self.waterfall_image.setRect(
                    x_min,
                    0,
                    x_max - x_min,
                    self._waterfall_rows,
                )

        self.waterfall_plot.setYRange(0, self._waterfall_rows, padding=0.0)

    def _show_blank_waterfall(self):
        """
        Initialise the waterfall as an all-black image.
        """

        self._waterfall_buffer = np.full(
            (self._waterfall_rows, 2),
            WATERFALL_EMPTY_VALUE,
            dtype=np.float32,
        )
        self._waterfall_frequency_range = None

        self._draw_waterfall()

        # No frequency data yet: give the blank image a unit-wide rect.
        self.waterfall_image.setRect(0, 0, 1, self._waterfall_rows)

    def _update_waterfall(
        self,
        powers: np.ndarray,
        x_min: float,
        x_max: float,
    ):

        row = np.asarray(powers, dtype=np.float32)

        if row.size == 0:
            return

        # First frame or changed FFT size
        if (
            self._waterfall_buffer is None
            or self._waterfall_buffer.shape[1] != row.size
        ):
            self._waterfall_buffer = np.full(
                (self._waterfall_rows, row.size),
                WATERFALL_EMPTY_VALUE,
                dtype=np.float32,
            )

        # Move old rows DOWN
        self._waterfall_buffer[1:] = self._waterfall_buffer[:-1]

        # Newest row goes at the TOP.
        self._waterfall_buffer[0] = row

        self._waterfall_frequency_range = (x_min, x_max)

        self._draw_waterfall()

    # ==================================================================
    # Beacon markers
    # ==================================================================

    def _update_beacon_marker_visibility(self, x_min: float, x_max: float):

        for frequency, marker, beacon, band in zip(
            self._beacon_frequencies,
            self.beacon_markers,
            self._beacons,
            self.beacon_band_items,
        ):
            marker.setVisible(x_min <= frequency <= x_max)

            if band is not None:
                half_bandwidth = float(beacon["bandwidth"]) / 2.0
                band_start = frequency - half_bandwidth
                band_end = frequency + half_bandwidth
                band.setVisible(band_end >= x_min and band_start <= x_max)

    # ==================================================================
    # Frequency ranges
    # ==================================================================

    def _update_frequency_range_visibility(self, x_min: float, x_max: float):

        for frequency_range, items in zip(
            self._frequency_ranges,
            self._frequency_range_items,
        ):
            start = float(frequency_range.start_freq)
            end = float(frequency_range.end_freq)

            # A range is visible if any part of it
            # intersects the current FFT window.
            visible = end >= x_min and start <= x_max

            for item in items:
                item.setVisible(visible)

    # ==================================================================
    # View actions
    # ==================================================================

    def _reset_fft_view(self):

        if self._waterfall_frequency_range is None:
            return

        x_min, x_max = self._waterfall_frequency_range

        self.fft_plot.setXRange(x_min, x_max, padding=0.0)

    def _clear_waterfall(self):

        # Back to an all-black waterfall.
        if self._waterfall_buffer is not None:
            self._waterfall_buffer.fill(WATERFALL_EMPTY_VALUE)
            self._draw_waterfall()

    def _clear_max_hold(self):

        self._max_hold_values = None
        self.max_hold_curve.clear()

    # ==================================================================
    # File actions
    # ==================================================================

    def _export_beacons(self):
        """
        Ask where to save the beacon list and tell main.py.

        main.py receives the chosen file path through
        export_beacons_requested and writes the file.
        """

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Beacons",
            "beacons.csv",
            "CSV files (*.csv);;Text files (*.txt);;All files (*)",
        )

        if path:
            self.export_beacons_requested.emit(path)

    def _edit_ranges_placeholder(self):
        pass

    # ==================================================================
    # Formatting
    # ==================================================================

    @staticmethod
    def _format_frequency(frequency_hz: float) -> str:

        frequency_hz = float(frequency_hz)

        if abs(frequency_hz) >= 1e9:
            return f"{frequency_hz / 1e9:.6f} GHz"

        if abs(frequency_hz) >= 1e6:
            return f"{frequency_hz / 1e6:.6f} MHz"

        if abs(frequency_hz) >= 1e3:
            return f"{frequency_hz / 1e3:.3f} kHz"

        return f"{frequency_hz:.1f} Hz"