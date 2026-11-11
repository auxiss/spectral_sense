"""
range_selector.py

The "Range" drop-down of the scanner's Frequency box.

    - lists the known frequency ranges (known_ranges.csv); picking one fills
      Start and Stop
    - "Custom" is shown whenever Start / Stop do not match a known range
    - "Save..." (only while Custom is shown) asks for a name and saves the
      current Start / Stop as a new known range

The widget only reports what the user did; ScannerUI fills the Start / Stop
controls and scanner_main saves the range (save_custom_range below).
"""

from __future__ import annotations

from typing import Sequence

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

from known_ranges import rf_range


CUSTOM_TEXT = "Custom"
MATCH_TOLERANCE_HZ = 1.0            # Start / Stop this close to a known range count as that range

# Ranges saved from the scanner (valid_golbe_region has no meaning for them)
CUSTOM_REGION = "all"
CUSTOM_RANGE_COLORS = [
    "#00BFFF", "#FFA500", "#ADFF2F", "#FF69B4",
    "#BA55D3", "#40E0D0", "#FFD700", "#F08080",
]


def _format(frequency_hz: float) -> str:
    frequency_hz = float(frequency_hz)

    if abs(frequency_hz) >= 1e9:
        return f"{frequency_hz / 1e9:g} GHz"
    if abs(frequency_hz) >= 1e6:
        return f"{frequency_hz / 1e6:g} MHz"
    if abs(frequency_hz) >= 1e3:
        return f"{frequency_hz / 1e3:g} kHz"
    return f"{frequency_hz:g} Hz"


class RangeSelector(QWidget):
    range_chosen = pyqtSignal(float, float)            # start_hz, stop_hz
    save_requested = pyqtSignal(str, float, float)     # name, start_hz, stop_hz

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ranges = []
        self._start_hz = 0.0
        self._stop_hz = 0.0

        self.combo = QComboBox()
        self.combo.addItem(CUSTOM_TEXT, None)

        self.save_button = QPushButton("Save...")
        self.save_button.setToolTip("Save the current Start / Stop as a known range")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, stretch=1)
        layout.addWidget(self.save_button)

        # `activated` fires for the user's choice only, never for changes made by the program.
        self.combo.activated.connect(self._on_activated)
        self.save_button.clicked.connect(self._on_save_clicked)

        self._refresh_save_button()

    # ------------------------------------------------------------------
    # Called by ScannerUI
    # ------------------------------------------------------------------

    def set_ranges(self, ranges: Sequence):
        """Replace the list of known ranges (rf_range objects)."""

        self._ranges = sorted(
            (r for r in ranges if float(r.end_freq) > float(r.start_freq)),
            key=lambda r: float(r.start_freq),
        )

        blocked = self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(CUSTOM_TEXT, None)

        for known in self._ranges:
            start, end = float(known.start_freq), float(known.end_freq)
            self.combo.addItem(
                f"{known.info}   ({_format(start)} - {_format(end)})", (start, end)
            )

        self.combo.blockSignals(blocked)
        self.update_for(self._start_hz, self._stop_hz)

    def update_for(self, start_hz: float, stop_hz: float):
        """Start / Stop changed: show the matching known range, or Custom."""

        self._start_hz = float(start_hz)
        self._stop_hz = float(stop_hz)

        index = 0

        for i in range(1, self.combo.count()):
            start, end = self.combo.itemData(i)
            if (abs(start - self._start_hz) <= MATCH_TOLERANCE_HZ
                    and abs(end - self._stop_hz) <= MATCH_TOLERANCE_HZ):
                index = i
                break

        blocked = self.combo.blockSignals(True)
        self.combo.setCurrentIndex(index)
        self.combo.blockSignals(blocked)

        self._refresh_save_button()

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def _on_activated(self, index):
        data = self.combo.itemData(index)

        if data is None:                      # "Custom": keep whatever Start / Stop are
            return

        self.range_chosen.emit(float(data[0]), float(data[1]))

    def _refresh_save_button(self):
        self.save_button.setEnabled(
            self.combo.currentIndex() == 0 and self._stop_hz > self._start_hz
        )

    def _on_save_clicked(self):
        if self._stop_hz <= self._start_hz:
            return

        name, ok = QInputDialog.getText(
            self,
            "Save range",
            f"Name for {_format(self._start_hz)} - {_format(self._stop_hz)}:",
        )
        name = name.strip()

        if not ok or not name:
            return

        if any(str(r.info).strip().lower() == name.lower() for r in self._ranges):
            answer = QMessageBox.question(
                self, "Save range", f"A range called '{name}' already exists.\nReplace it?"
            )
            if answer != QMessageBox.Yes:
                return

        self.save_requested.emit(name, self._start_hz, self._stop_hz)


# ----------------------------------------------------------------------
# Saving (used by scanner_main)
# ----------------------------------------------------------------------

def save_custom_range(manager, info: str, start_hz: float, stop_hz: float) -> rf_range:
    """
    Add a range to the known ranges (ranges_manager) and save the file.
    A range with the same name is replaced and keeps its id and colour.
    """

    info = info.strip()
    ranges = manager.get_ranges()

    existing = next(
        (r for r in ranges if str(r.info).strip().lower() == info.lower()), None
    )

    if existing is not None:
        new = rf_range(
            id=existing.id,
            start_freq=float(start_hz),
            end_freq=float(stop_hz),
            info=info,
            valid_golbe_region=existing.valid_golbe_region,
            hex_color=existing.hex_color,
        )
        manager.replace(new)
        return new

    new_id = max((r.id for r in ranges), default=-1) + 1

    new = rf_range(
        id=new_id,
        start_freq=float(start_hz),
        end_freq=float(stop_hz),
        info=info,
        valid_golbe_region=CUSTOM_REGION,
        hex_color=CUSTOM_RANGE_COLORS[new_id % len(CUSTOM_RANGE_COLORS)],
    )
    manager.add(new)
    return new