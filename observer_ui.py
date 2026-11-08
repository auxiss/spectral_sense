"""
observer_ui.py

Window for the observing mode (same look as scanner_ui.py).

Layout (left half = plots, right half = controls):

    +---------------------------+---------------------------+
    | wideband spectrum (thin)  |  CONTROL PANEL            |
    |  + marker of the channel  |   source / mode / target  |
    | IQ      | channel spectrum|   gain / threshold / start|
    | density |  (+ threshold)  |   channel edit | recordings|
    |---------| channel waterfall---------------------------+
    |features | (x axes linked, |  CHANNEL LIST             |
    | list    |  centred on 0)  |   select all demod/record |
    | [disk scroll bar]         |                           |
    +---------------------------+---------------------------+

The UI only displays and reports. All work is done by observer_app.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import features as feat
from scanner_ui import (
    AXIS_COLOR,
    BACKGROUND_COLOR,
    BORDER_COLOR,
    DISABLED_TEXT_COLOR,
    FFT_COLOR,
    GRID_ALPHA,
    PANEL_COLOR,
    TEXT_COLOR,
    THRESHOLD_COLOR,
    WATERFALL_COLOR_STOPS,
    WINDOW_BACKGROUND_COLOR,
)


# ======================================================================
# Colours and display constants (edit here)
# ======================================================================

OBSERVED_COLOR = "#FFAA00"          # marker of the observed channel on the wideband plot
OBSERVED_BAND_ALPHA = 50            # its fill opacity, 0 to 255

ACTIVE_ROW_COLOR = "#1F4D1F"        # channel list row: currently transmitting
CURRENT_ROW_COLOR = "#33334D"       # channel list row: currently observed

WIDEBAND_HEIGHT = 130               # the wideband plot is wide and thin
LEFT_AXIS_WIDTH = 55                # same on spectrum and waterfall so their x axes line up
ROW_HEIGHT = 44                     # channel list rows
IQ_EXTENT = feat.IQ_DENSITY_EXTENT


class _SquareHolder(QWidget):
    """Keeps its child square and centred in whatever space it is given."""

    def __init__(self, child):
        super().__init__()
        self._child = child
        child.setParent(self)
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return width                       # as tall as it is wide

    def resizeEvent(self, event):
        side = min(self.width(), self.height())
        self._child.setGeometry(
            (self.width() - side) // 2, (self.height() - side) // 2, side, side
        )
        super().resizeEvent(event)


class ObserverUI(QMainWindow):
    # ------------------------------------------------------------------
    # Signals: UI -> application
    # ------------------------------------------------------------------

    source_changed = pyqtSignal(str)               # "live" / "disk"
    mode_changed = pyqtSignal(str)                 # "manual" / "auto"
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    target_samples_changed = pyqtSignal(int)
    gain_changed = pyqtSignal(int)
    threshold_changed = pyqtSignal(float)

    channel_selected = pyqtSignal(int)             # row clicked
    demod_ids_changed = pyqtSignal(object)         # set of channel ids
    record_ids_changed = pyqtSignal(object)        # set of channel ids

    channel_edit_saved = pyqtSignal(int, object)   # id, dict(name,start,end,modulation,protocol,info)
    channel_delete_requested = pyqtSignal(int)
    modulation_text_changed = pyqtSignal(str)      # so the app can fill the protocol list

    capture_selected = pyqtSignal(int, str)        # channel id, capture file name
    capture_delete_requested = pyqtSignal(int, str)
    disk_position_changed = pyqtSignal(int)        # first sample of the shown window

    audio_window_requested = pyqtSignal()          # View > Audio playback
    text_window_requested = pyqtSignal()           # View > Text output (placeholder, nothing listens yet)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("SDR Observer")
        self.resize(1500, 850)

        self._channels = []
        self._decodable = set()
        self._demod_ids = set()
        self._record_ids = set()
        self._active_ids = set()
        self._collected = {}
        self._target = 0

        self._selected_id = None
        self._current_id = None
        self._edit_id = None

        self._live = True
        self._running = False
        self._x_range = None

        self._loading = False             # edit area is being filled by the program
        self._edit_touched = False        # the user changed something in the edit area
        self._geometry_touched = False    # ... centre or bandwidth in particular

        self._build_ui()
        self._build_menus()
        self._connect_internal_signals()
        self.clear_plots()
        self._update_enabled()

    # ==================================================================
    # PUBLIC UI API
    # ==================================================================

    # ---------------- configuration ----------------

    def set_configuration(self, *, gain: int, threshold: float,
                          target_samples: int, mode: str, source: str):
        """Set the controls without emitting signals."""

        controls = (self.gain_slider, self.threshold_spin, self.target_spin,
                    self.mode_combo, self.source_combo)
        blocked = [c.blockSignals(True) for c in controls]

        try:
            self.gain_slider.setValue(int(gain))
            self.gain_value_label.setText(f"{int(gain)}%")
            self.threshold_spin.setValue(float(threshold))
            self.target_spin.setValue(int(target_samples))
            self.mode_combo.setCurrentText(mode.capitalize())
            self.source_combo.setCurrentText(source.capitalize())
        finally:
            for control, was_blocked in zip(controls, blocked):
                control.blockSignals(was_blocked)

        self._target = int(target_samples)
        self._live = source == "live"
        self._apply_source_visuals()
        self._update_enabled()

    def set_running(self, active: bool):
        """
        While running in AUTO mode everything is locked except Stop.
        In manual mode the channel list, gain and threshold stay usable.
        """

        self._running = bool(active)
        self._update_enabled()

    def source(self) -> str:
        return "live" if self._live else "disk"

    def mode(self) -> str:
        return self.mode_combo.currentText().lower()

    # ---------------- channel list ----------------

    def set_channels(self, channels: Sequence, decodable_ids=None):
        """
        Replace the channel list. `channels` are ChannelManager.RFRange
        objects. decodable_ids = ids whose modulation has a demodulator
        (only those can be switched to demod).
        """

        self._channels = list(channels)
        ids = {c.id for c in self._channels}

        if decodable_ids is not None:
            self._decodable = set(decodable_ids)

        self._demod_ids &= ids & self._decodable
        self._record_ids &= ids
        self._active_ids &= ids

        if self._selected_id not in ids:
            self._selected_id = None
        if self._current_id not in ids:
            self._current_id = None
        if self._edit_id not in ids:
            self._clear_edit()

        self._rebuild_table()

        # Follow the worker's refinements, but never overwrite what is being typed.
        if self._edit_id is not None and not self._edit_touched:
            self._load_edit(self._edit_id)

    def set_current_channel(self, channel_id):
        """Channel the SDR is tuned to right now."""
        self._current_id = channel_id
        self._apply_row_colors()

    def set_channel_active(self, channel_id: int, active: bool):
        if active:
            self._active_ids.add(channel_id)
        else:
            self._active_ids.discard(channel_id)
        self._apply_row_colors()

    def set_progress(self, collected: dict, target: int):
        """Samples saved this run per channel (auto mode)."""
        self._collected = dict(collected)
        self._target = int(target)

        for row, channel in enumerate(self._channels):
            item = self.channel_table.item(row, 0)
            if item is not None:
                item.setText(self._row_text(channel))

    def select_channel(self, channel_id):
        """Select a row programmatically (no signal)."""
        self._selected_id = channel_id

        for row, channel in enumerate(self._channels):
            if channel.id == channel_id:
                blocked = self.channel_table.blockSignals(True)
                self.channel_table.selectRow(row)
                self.channel_table.blockSignals(blocked)

    def selected_channel_id(self):
        return self._selected_id

    def demod_ids(self) -> set:
        return set(self._demod_ids)

    def record_ids(self) -> set:
        return set(self._record_ids)

    # ---------------- edit area ----------------

    def set_modulation_choices(self, names: Sequence[str]):
        self._set_combo_items(self.modulation_combo, names)

    def set_protocol_choices(self, names: Sequence[str]):
        self._set_combo_items(self.protocol_combo, names)

    # ---------------- recordings strip ----------------

    def set_captures(self, rows: Sequence[dict]):
        """rows = ChannelManager.get_captures() dicts."""

        self.capture_list.clear()

        for row in rows:
            try:
                rate = float(row["bandwidth"])
                text = (
                    f"{row['capture_file']}   {str(row['time'])[11:19]}   "
                    f"{int(float(row['num_samples'])):,} smp @ {self._format_rate(rate)}"
                )
            except (KeyError, ValueError):
                text = str(row.get("capture_file", "?"))

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, row.get("capture_file"))
            self.capture_list.addItem(item)

    # ---------------- plots ----------------

    def set_wideband(self, frequencies_hz, power_db):
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)

        if frequencies.size == 0:
            return

        self.wideband_curve.setData(frequencies, np.asarray(power_db, dtype=np.float32))
        self.wideband_plot.setXRange(
            float(frequencies.min()), float(frequencies.max()), padding=0.0
        )

    def set_channel_marker(self, start_hz: float, end_hz: float):
        """Shaded band + centre line of the observed channel on the wideband plot."""

        if not self._live:
            return

        self.channel_band.setRegion((float(start_hz), float(end_hz)))
        self.channel_line.setValue((float(start_hz) + float(end_hz)) / 2.0)
        self.channel_band.setVisible(True)
        self.channel_line.setVisible(True)

    def set_features(self, data: dict):
        """
        Show one result of features.ChannelWindow.compute() /
        features_from_samples(). Optional keys: gain_db, bandwidth.
        """

        frequencies = np.asarray(data["frequencies"], dtype=np.float64)
        spectrum = np.asarray(data["spectrum_db"], dtype=np.float32)
        waterfall = np.asarray(data["waterfall_db"], dtype=np.float32)

        x_min, x_max = float(frequencies.min()), float(frequencies.max())

        # Channel spectrum
        self.spectrum_curve.setData(frequencies, spectrum)

        if self._x_range != (x_min, x_max) and x_max > x_min:
            self._x_range = (x_min, x_max)
            self.spectrum_plot.setXRange(x_min, x_max, padding=0.0)   # waterfall follows (linked)

        # Waterfall: row 0 is the newest row. Fewer rows (short recording)
        # just fill the top part.
        rows = waterfall.shape[0]

        self.waterfall_image.setImage(
            waterfall,
            autoLevels=False,
            levels=data["levels"],
            axisOrder="row-major",
        )
        self.waterfall_image.setRect(x_min, 0, x_max - x_min, rows)
        self.waterfall_plot.setYRange(0, feat.WATERFALL_ROWS, padding=0.0)

        # IQ density
        self.iq_image.setImage(
            np.asarray(data["iq_density"], dtype=np.float32),
            autoLevels=False,
            levels=(0.0, 1.0),
            axisOrder="row-major",
        )
        self.iq_image.setRect(-IQ_EXTENT, -IQ_EXTENT, 2 * IQ_EXTENT, 2 * IQ_EXTENT)

        # Threshold line (live only: it needs the noise floor next to the channel)
        level = data.get("threshold_level_db")

        if level is None:
            self.threshold_line.setVisible(False)
        else:
            self.threshold_line.setValue(float(level))
            self.threshold_line.setVisible(True)

        # Features list (gain, bandwidth, sample rate are only displayed)
        self._set_metrics(
            data.get("gain_db"),
            data.get("snr_db"),
            data.get("bandwidth"),
            data.get("sample_rate"),
        )

    def clear_channel_plots(self):
        self.spectrum_curve.clear()
        self.waterfall_image.clear()
        self.iq_image.clear()
        self._x_range = None
        self.threshold_line.setVisible(False)
        self._set_metrics(None, None, None, None)

    def clear_plots(self):
        self.wideband_curve.clear()
        self.channel_band.setVisible(False)
        self.channel_line.setVisible(False)
        self.clear_channel_plots()

    def set_disk_recording(self, total_samples: int):
        """Size of the loaded recording: sets the range of the scroll bar."""

        blocked = self.disk_slider.blockSignals(True)
        self.disk_slider.setRange(0, max(0, int(total_samples) - feat.WINDOW_SAMPLES))
        self.disk_slider.setValue(0)
        self.disk_slider.blockSignals(blocked)

        self.disk_slider.setEnabled(self.disk_slider.maximum() > 0)
        self._update_disk_label(0, total_samples)
        self._disk_total = int(total_samples)

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        main_layout.addWidget(self._build_plots(), stretch=1)
        main_layout.addWidget(self._build_right_half(), stretch=1)

    def _build_menus(self):
        view_menu = self.menuBar().addMenu("View")

        audio_action = QAction("Audio playback", self)
        text_action = QAction("Text output", self)

        audio_action.triggered.connect(self.audio_window_requested.emit)
        text_action.triggered.connect(self.text_window_requested.emit)   # later: decoded text window

        view_menu.addAction(audio_action)
        view_menu.addAction(text_action)

    # ---------------- left half: plots ----------------

    def _build_plots(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Wideband (full width, thin) -----------------------------------
        self.wideband_plot = pg.PlotWidget()
        self._style_plot(self.wideband_plot)
        self.wideband_plot.setFixedHeight(WIDEBAND_HEIGHT)
        self.wideband_plot.setLabel("left", "Power", units="dB", color=TEXT_COLOR)
        self.wideband_plot.setLabel("bottom", "Frequency", units="Hz", color=TEXT_COLOR)
        self.wideband_plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.wideband_plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)

        self.wideband_curve = self.wideband_plot.plot(pen=pg.mkPen(FFT_COLOR, width=1))

        band_color = QColor(OBSERVED_COLOR)
        band_color.setAlpha(OBSERVED_BAND_ALPHA)

        self.channel_band = pg.LinearRegionItem(
            values=(0, 1),
            orientation="vertical",
            movable=False,
            brush=QBrush(band_color),
            pen=pg.mkPen(OBSERVED_COLOR, width=0.5),
        )
        self.channel_band.setZValue(-10)
        self.channel_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(OBSERVED_COLOR, width=1, style=Qt.DashLine),
        )
        self.wideband_plot.addItem(self.channel_band)
        self.wideband_plot.addItem(self.channel_line)

        layout.addWidget(self.wideband_plot, stretch=0)

        # IQ density | channel spectrum over channel waterfall ----------
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)

        self.iq_plot = pg.PlotWidget()
        self._style_plot(self.iq_plot)
        self.iq_plot.setLabel("left", "Q", color=TEXT_COLOR)
        self.iq_plot.setLabel("bottom", "I", color=TEXT_COLOR)
        self.iq_plot.setXRange(-IQ_EXTENT, IQ_EXTENT, padding=0.0)
        self.iq_plot.setYRange(-IQ_EXTENT, IQ_EXTENT, padding=0.0)
        self.iq_plot.setMouseEnabled(x=False, y=False)
        self.iq_plot.getViewBox().disableAutoRange()
        self.iq_plot.setMenuEnabled(False)

        self.iq_image = pg.ImageItem()
        self.iq_image.setLookupTable(self._lookup_table())
        self.iq_plot.addItem(self.iq_image)

        self.spectrum_plot = pg.PlotWidget()
        self._style_plot(self.spectrum_plot)
        self.spectrum_plot.setLabel("left", "Power", units="dB", color=TEXT_COLOR)
        self.spectrum_plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.spectrum_plot.hideAxis("bottom")               # the waterfall shows the shared axis
        self.spectrum_plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(FFT_COLOR, width=1))

        self.threshold_line = pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(THRESHOLD_COLOR, width=1, style=Qt.DashLine),
        )
        self.spectrum_plot.addItem(self.threshold_line)
        self.threshold_line.setVisible(False)

        self.waterfall_plot = pg.PlotWidget()
        self._style_plot(self.waterfall_plot)
        self.waterfall_plot.invertY()
        self.waterfall_plot.setXLink(self.spectrum_plot)    # same frequency axis
        self.waterfall_plot.setLabel("left", "Time", color=TEXT_COLOR)
        self.waterfall_plot.setLabel("bottom", "Frequency offset", units="Hz", color=TEXT_COLOR)
        self.waterfall_plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.waterfall_plot.getAxis("left").setStyle(showValues=False)

        self.waterfall_image = pg.ImageItem()
        self.waterfall_image.setLookupTable(self._lookup_table())
        self.waterfall_plot.addItem(self.waterfall_image)

        # Left column: constellation with the features list right below it,
        # exactly as wide as the constellation.
        features_box = QGroupBox("Features")
        features_form = QFormLayout(features_box)
        self.metric_labels = {}

        for name in ("Gain", "SNR", "Bandwidth", "Sample rate"):
            label = QLabel("--")
            features_form.addRow(f"{name}:", label)
            self.metric_labels[name] = label

        iq_column = QWidget()
        iq_layout = QVBoxLayout(iq_column)
        iq_layout.setContentsMargins(0, 0, 0, 0)
        iq_layout.addWidget(_SquareHolder(self.iq_plot), stretch=0)
        iq_layout.addWidget(features_box, stretch=0)          # directly under the constellation
        iq_layout.addStretch(1)

        grid.addWidget(iq_column, 0, 0, 2, 1)
        grid.addWidget(self.spectrum_plot, 0, 1)
        grid.addWidget(self.waterfall_plot, 1, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 3)

        layout.addLayout(grid, stretch=1)

        # Disk scroll bar (disk mode only) ------------------------------
        self.disk_widget = QWidget()
        disk_layout = QHBoxLayout(self.disk_widget)
        disk_layout.setContentsMargins(0, 0, 0, 0)

        self.disk_slider = QSlider(Qt.Horizontal)
        self.disk_slider.setRange(0, 0)
        self.disk_label = QLabel("")
        self._disk_total = 0

        disk_layout.addWidget(self.disk_slider, stretch=1)
        disk_layout.addWidget(self.disk_label)

        layout.addWidget(self.disk_widget, stretch=0)

        return widget

    # ---------------- right half: control panel + channel list ----------------

    def _build_right_half(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_control_panel(), stretch=1)
        layout.addWidget(self._build_channel_list(), stretch=1)

        return widget

    def _build_control_panel(self):
        self.control_group = QGroupBox("Controls")
        layout = QVBoxLayout(self.control_group)

        # --- top block: source, mode, target, gain, threshold, start/stop
        top = QGridLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Live", "Disk"])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Auto"])

        self.target_spin = QSpinBox()
        self.target_spin.setRange(1024, 2_000_000_000)
        self.target_spin.setSingleStep(65536)
        self.target_spin.setValue(1_000_000)
        self.target_spin.setSuffix(" samples")

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(20)
        self.gain_value_label = QLabel("20%")

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 60.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setSuffix(" dB")
        self.threshold_spin.setValue(6.0)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        gain_widget = QWidget()
        gain_layout = QHBoxLayout(gain_widget)
        gain_layout.setContentsMargins(0, 0, 0, 0)
        gain_layout.addWidget(self.gain_slider)
        gain_layout.addWidget(self.gain_value_label)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)

        top.addWidget(QLabel("Source:"), 0, 0)
        top.addWidget(self.source_combo, 0, 1)
        top.addWidget(QLabel("Mode:"), 0, 2)
        top.addWidget(self.mode_combo, 0, 3)
        top.addWidget(QLabel("Target:"), 0, 4)
        top.addWidget(self.target_spin, 0, 5)

        top.addWidget(QLabel("Gain:"), 1, 0)
        top.addWidget(gain_widget, 1, 1)
        top.addWidget(QLabel("Threshold:"), 1, 2)
        top.addWidget(self.threshold_spin, 1, 3)
        top.addWidget(buttons, 1, 4, 1, 2)

        layout.addLayout(top)

        # --- bottom block: channel edit | recordings
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_edit_area(), stretch=1)
        bottom.addWidget(self._build_captures_area(), stretch=1)
        layout.addLayout(bottom, stretch=1)

        return self.control_group

    def _build_edit_area(self):
        self.edit_group = QGroupBox("Channel")
        form = QFormLayout(self.edit_group)

        self.name_edit = QLineEdit()

        self.center_spin = self._make_frequency_spinbox()
        self.center_unit = self._make_unit_combo()
        self.bandwidth_spin = self._make_frequency_spinbox()
        self.bandwidth_unit = self._make_unit_combo()

        self.modulation_combo = QComboBox()
        self.modulation_combo.setEditable(True)           # type a new name to create one
        self.protocol_combo = QComboBox()
        self.protocol_combo.setEditable(True)

        self.info_edit = QLineEdit()

        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete channel")

        form.addRow("Name:", self.name_edit)
        form.addRow("Center:", self._make_value_unit_widget(self.center_spin, self.center_unit))
        form.addRow("Bandwidth:", self._make_value_unit_widget(self.bandwidth_spin, self.bandwidth_unit))
        form.addRow("Modulation:", self.modulation_combo)
        form.addRow("Protocol:", self.protocol_combo)
        form.addRow("Info:", self.info_edit)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.save_button)
        row_layout.addWidget(self.delete_button)
        form.addRow(row)

        return self.edit_group

    def _build_captures_area(self):
        self.captures_group = QGroupBox("Recordings")
        layout = QVBoxLayout(self.captures_group)

        self.capture_list = QListWidget()
        self.capture_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.delete_capture_button = QPushButton("Delete recording")

        layout.addWidget(self.capture_list, stretch=1)
        layout.addWidget(self.delete_capture_button)

        return self.captures_group

    def _build_channel_list(self):
        self.channels_group = QGroupBox("Channels")
        layout = QVBoxLayout(self.channels_group)

        header = QHBoxLayout()
        self.select_all_demod_checkbox = QCheckBox("Select all demod")
        self.select_all_record_checkbox = QCheckBox("Select all record")
        header.addWidget(self.select_all_demod_checkbox)
        header.addWidget(self.select_all_record_checkbox)
        header.addStretch()
        layout.addLayout(header)

        self.channel_table = QTableWidget(0, 3)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Demod", "Record"])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.channel_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.channel_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.channel_table.setShowGrid(False)

        header_view = self.channel_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeToContents)

        layout.addWidget(self.channel_table)

        return self.channels_group

    def _apply_theme(self):
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

            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
            QLineEdit, QListWidget, QTableWidget {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 2px;
            }}

            QPushButton:checked {{
                background-color: {BORDER_COLOR};
                border: 1px solid {TEXT_COLOR};
            }}

            QHeaderView::section {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 2px;
            }}

            QTableWidget::item:selected, QListWidget::item:selected {{
                background-color: {BORDER_COLOR};
                color: {TEXT_COLOR};
            }}

            QComboBox QAbstractItemView {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                selection-background-color: {BORDER_COLOR};
            }}

            QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled,
            QDoubleSpinBox:disabled, QLineEdit:disabled, QLabel:disabled,
            QCheckBox:disabled, QGroupBox:disabled, QListWidget:disabled,
            QTableWidget:disabled {{
                color: {DISABLED_TEXT_COLOR};
            }}
        """)

    @staticmethod
    def _style_plot(plot):
        plot.setBackground(BACKGROUND_COLOR)

        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(AXIS_COLOR))
            axis.setTextPen(pg.mkPen(TEXT_COLOR))

    @staticmethod
    def _lookup_table():
        """Same colour ramp as the scanner waterfall."""
        colour_map = pg.ColorMap(
            [position for position, _ in WATERFALL_COLOR_STOPS],
            [colour for _, colour in WATERFALL_COLOR_STOPS],
        )
        return colour_map.getLookupTable(0.0, 1.0, 256)

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _connect_internal_signals(self):
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)

        self.target_spin.valueChanged.connect(self._on_target_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.threshold_spin.valueChanged.connect(self.threshold_changed.emit)

        self.channel_table.cellClicked.connect(self._on_row_clicked)
        self.select_all_demod_checkbox.clicked.connect(self._on_select_all_demod)
        self.select_all_record_checkbox.clicked.connect(self._on_select_all_record)

        self.save_button.clicked.connect(self._on_save_clicked)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        self.modulation_combo.currentTextChanged.connect(self.modulation_text_changed.emit)

        # What the user touched in the edit area (programmatic loads do not count)
        self.name_edit.textEdited.connect(self._touch)
        self.info_edit.textEdited.connect(self._touch)
        self.modulation_combo.currentTextChanged.connect(self._touch)
        self.protocol_combo.currentTextChanged.connect(self._touch)

        for widget in (self.center_spin, self.bandwidth_spin):
            widget.valueChanged.connect(self._touch_geometry)
        for widget in (self.center_unit, self.bandwidth_unit):
            widget.currentTextChanged.connect(self._touch_geometry)

        self.capture_list.itemClicked.connect(self._on_capture_clicked)
        self.delete_capture_button.clicked.connect(self._on_delete_capture_clicked)

        self.disk_slider.valueChanged.connect(self._on_disk_slider)

    # ==================================================================
    # Enable / visibility rules
    # ==================================================================

    def _update_enabled(self):
        live = self._live
        running = self._running
        auto = self.mode_combo.currentText() == "Auto"
        locked = running and auto

        self.source_combo.setEnabled(not running)
        self.mode_combo.setEnabled(live and not running)
        self.target_spin.setEnabled(live and auto and not running)

        self.start_button.setEnabled(live and not running)
        self.stop_button.setEnabled(running)            # Stop is always available while running

        self.gain_slider.setEnabled(live and not locked)
        self.threshold_spin.setEnabled(live and not locked)

        self.edit_group.setEnabled(not locked)
        self.captures_group.setEnabled(not locked)
        self.channels_group.setEnabled(not locked)

    def _apply_source_visuals(self):
        self.disk_widget.setVisible(not self._live)

    # ==================================================================
    # Control handlers
    # ==================================================================

    def _on_source_changed(self, text):
        self._live = text.lower() == "live"
        self._apply_source_visuals()
        self.clear_plots()
        self._update_enabled()
        self.source_changed.emit(text.lower())

    def _on_mode_changed(self, text):
        self._update_enabled()
        self.mode_changed.emit(text.lower())

    def _on_target_changed(self, value):
        self._target = int(value)
        self.target_samples_changed.emit(int(value))

    def _on_gain_changed(self, value):
        self.gain_value_label.setText(f"{value}%")
        self.gain_changed.emit(int(value))

    def _on_disk_slider(self, value):
        self._update_disk_label(value, self._disk_total)
        self.disk_position_changed.emit(int(value))

    def _update_disk_label(self, position, total):
        self.disk_label.setText(f"{int(position):,} / {int(total):,}")

    # ==================================================================
    # Channel list
    # ==================================================================

    def _row_text(self, channel) -> str:
        center = (channel.start_freq + channel.end_freq) / 2.0
        bandwidth = channel.end_freq - channel.start_freq

        line = f"{self._format_frequency(center)}   BW {self._format_frequency(bandwidth)}   {channel.modulation}"

        if channel.protocol not in ("", "unknown"):
            line += f" / {channel.protocol}"

        title = channel.name

        collected = self._collected.get(channel.id)
        if collected:
            title += f"   [{collected:,}"
            title += f" / {self._target:,}]" if self._target else "]"

        return f"{title}\n{line}"

    def _rebuild_table(self):
        blocked = self.channel_table.blockSignals(True)
        self.channel_table.setRowCount(len(self._channels))

        for row, channel in enumerate(self._channels):
            item = QTableWidgetItem(self._row_text(channel))
            item.setData(Qt.UserRole, channel.id)
            self.channel_table.setItem(row, 0, item)

            demod = self._make_toggle_button("Demod")
            demod.setEnabled(channel.id in self._decodable)
            demod.setChecked(channel.id in self._demod_ids)
            demod.toggled.connect(lambda checked, cid=channel.id: self._on_demod_toggled(cid, checked))

            record = self._make_toggle_button("Record")
            record.setChecked(channel.id in self._record_ids)
            record.toggled.connect(lambda checked, cid=channel.id: self._on_record_toggled(cid, checked))

            self.channel_table.setCellWidget(row, 1, demod)
            self.channel_table.setCellWidget(row, 2, record)

            if channel.id == self._selected_id:
                self.channel_table.selectRow(row)

        self.channel_table.blockSignals(blocked)

        self._apply_row_colors()
        self._sync_select_all()

    @staticmethod
    def _make_toggle_button(text):
        button = QPushButton(text)
        button.setCheckable(True)
        return button

    def _apply_row_colors(self):
        for row, channel in enumerate(self._channels):
            item = self.channel_table.item(row, 0)
            if item is None:
                continue

            if channel.id in self._active_ids:
                item.setBackground(QBrush(QColor(ACTIVE_ROW_COLOR)))
            elif channel.id == self._current_id:
                item.setBackground(QBrush(QColor(CURRENT_ROW_COLOR)))
            else:
                item.setBackground(QBrush())

            font = item.font()
            font.setBold(channel.id == self._current_id)
            item.setFont(font)

    def _on_row_clicked(self, row, _column):
        item = self.channel_table.item(row, 0)
        if item is None:
            return

        self._selected_id = int(item.data(Qt.UserRole))
        self._load_edit(self._selected_id)               # clicking a channel loads it in the control panel
        self.channel_selected.emit(self._selected_id)

    def _on_demod_toggled(self, channel_id, checked):
        if checked:
            self._demod_ids.add(channel_id)
        else:
            self._demod_ids.discard(channel_id)

        self._sync_select_all()
        self.demod_ids_changed.emit(set(self._demod_ids))

    def _on_record_toggled(self, channel_id, checked):
        if checked:
            self._record_ids.add(channel_id)
        else:
            self._record_ids.discard(channel_id)

        self._sync_select_all()
        self.record_ids_changed.emit(set(self._record_ids))

    def _set_column_checked(self, column, ids):
        for row, channel in enumerate(self._channels):
            button = self.channel_table.cellWidget(row, column)
            if button is None or not button.isEnabled():
                continue

            blocked = button.blockSignals(True)
            button.setChecked(channel.id in ids)
            button.blockSignals(blocked)

    def _on_select_all_demod(self, checked):
        self._demod_ids = (
            {c.id for c in self._channels if c.id in self._decodable} if checked else set()
        )
        self._set_column_checked(1, self._demod_ids)
        self.demod_ids_changed.emit(set(self._demod_ids))

    def _on_select_all_record(self, checked):
        self._record_ids = {c.id for c in self._channels} if checked else set()
        self._set_column_checked(2, self._record_ids)
        self.record_ids_changed.emit(set(self._record_ids))

    def _sync_select_all(self):
        """Keep the two header check boxes in step with the row buttons."""

        eligible_demod = {c.id for c in self._channels if c.id in self._decodable}
        all_ids = {c.id for c in self._channels}

        for checkbox, selected, universe in (
            (self.select_all_demod_checkbox, self._demod_ids, eligible_demod),
            (self.select_all_record_checkbox, self._record_ids, all_ids),
        ):
            blocked = checkbox.blockSignals(True)
            checkbox.setChecked(bool(universe) and selected >= universe)
            checkbox.blockSignals(blocked)

    # ==================================================================
    # Channel edit area
    # ==================================================================

    def _load_edit(self, channel_id):
        channel = next((c for c in self._channels if c.id == channel_id), None)

        if channel is None:
            self._clear_edit()
            return

        self._edit_id = channel_id

        center = (channel.start_freq + channel.end_freq) / 2.0
        bandwidth = channel.end_freq - channel.start_freq

        self._loading = True
        try:
            self.name_edit.setText(channel.name)
            self._set_frequency_control(self.center_spin, self.center_unit, center)
            self._set_frequency_control(self.bandwidth_spin, self.bandwidth_unit, bandwidth)

            blocked = self.modulation_combo.blockSignals(True)
            self.modulation_combo.setCurrentText(channel.modulation)
            self.modulation_combo.blockSignals(blocked)

            self.modulation_text_changed.emit(channel.modulation)    # fills the protocol list
            self.protocol_combo.setCurrentText(channel.protocol)
            self.info_edit.setText(channel.info)
        finally:
            self._loading = False

        self._edit_touched = False
        self._geometry_touched = False

    def _touch(self, *_):
        if not self._loading:
            self._edit_touched = True

    def _touch_geometry(self, *_):
        if not self._loading:
            self._edit_touched = True
            self._geometry_touched = True

    def _clear_edit(self):
        self._edit_id = None
        self.name_edit.clear()
        self.info_edit.clear()

    def _on_save_clicked(self):
        if self._edit_id is None:
            return

        center = self._to_hz(self.center_spin.value(), self.center_unit.currentText())
        bandwidth = self._to_hz(self.bandwidth_spin.value(), self.bandwidth_unit.currentText())

        if bandwidth <= 0:
            return

        # The range is only sent if the user changed it, so a stale value can
        # never overwrite what the observer has measured in the meantime.
        edited_range = self._geometry_touched
        self._edit_touched = False
        self._geometry_touched = False

        self.channel_edit_saved.emit(self._edit_id, {
            "name": self.name_edit.text().strip(),
            "start": center - bandwidth / 2.0 if edited_range else None,
            "end": center + bandwidth / 2.0 if edited_range else None,
            "modulation": self.modulation_combo.currentText().strip(),
            "protocol": self.protocol_combo.currentText().strip(),
            "info": self.info_edit.text().strip(),
        })

    def _on_delete_clicked(self):
        if self._edit_id is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete channel",
            "Delete this channel?\nIts recordings stay on disk.",
        )

        if answer == QMessageBox.Yes:
            self.channel_delete_requested.emit(self._edit_id)

    # ==================================================================
    # Recordings strip
    # ==================================================================

    def _on_capture_clicked(self, item):
        if self._selected_id is None:
            return

        capture_file = str(item.data(Qt.UserRole))

        if self._live:
            if self._running:
                return                      # the plots are showing the live feed: stop first

            self.source_combo.setCurrentText("Disk")     # idle: switch to disk mode, then load
            self._select_capture(capture_file)

        self.capture_selected.emit(self._selected_id, capture_file)

    def _select_capture(self, capture_file):
        for row in range(self.capture_list.count()):
            item = self.capture_list.item(row)
            if str(item.data(Qt.UserRole)) == capture_file:
                self.capture_list.setCurrentItem(item)
                return

    def _on_delete_capture_clicked(self):
        item = self.capture_list.currentItem()

        if item is None or self._selected_id is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete recording",
            "Delete this recording? This cannot be undone.",
        )

        if answer == QMessageBox.Yes:
            self.capture_delete_requested.emit(self._selected_id, str(item.data(Qt.UserRole)))

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _set_combo_items(combo, names):
        text = combo.currentText()
        blocked = combo.blockSignals(True)

        combo.clear()
        combo.addItems(list(names))
        combo.setCurrentText(text)

        combo.blockSignals(blocked)

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

    @staticmethod
    def _to_hz(value, unit):
        return float(value) * {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]

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

        multiplier = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[unit]

        unit_widget.setCurrentText(unit)
        value_widget.setValue(frequency_hz / multiplier)

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

    @staticmethod
    def _format_rate(rate_hz: float) -> str:
        if rate_hz >= 1e6:
            return f"{rate_hz / 1e6:.3f} MS/s"
        return f"{rate_hz / 1e3:.2f} kS/s"

    def _set_metrics(self, gain_db, snr_db, bandwidth, sample_rate):
        values = {
            "Gain": "--" if gain_db is None else f"{float(gain_db):.1f} dB",
            "SNR": "--" if snr_db is None else f"{float(snr_db):.1f} dB",
            "Bandwidth": "--" if bandwidth is None else self._format_frequency(bandwidth),
            "Sample rate": "--" if sample_rate is None else self._format_rate(float(sample_rate)),
        }

        for name, text in values.items():
            self.metric_labels[name].setText(text)
