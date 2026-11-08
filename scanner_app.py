import csv

from PyQt5.QtCore import QObject, QThread, pyqtSlot

import known_ranges
from scanner_worker import ScannerWorker


class ScannerApplication(QObject):
    def __init__(self, ui, channels, show=True):
        super().__init__()
        self.ui = ui
        self.channels = channels
        self.last_beacons = []

        ranges = known_ranges.ranges_manager("no_file").get_ranges()
        self.ui.set_frequency_ranges(ranges)

        self.sample_rate = 2.4e6
        self.min_freq = 60e6
        self.max_freq = 150e6
        self.smothnes = 100
        self.gain = 20
        self.threshold = 2
        self.beacon_bandwidth = 100e3
        self.dynamic_thershold = True
        self.outer_cut = 700e3
        self.dc_spike_cut = 10e3
        self.neighbors = 100

        self.thread = None
        self.worker = None

        self.ui.start_requested.connect(self.start)
        self.ui.stop_requested.connect(self.stop)
        self.ui.single_shot_requested.connect(self.single_shot)
        self.ui.export_beacons_requested.connect(self._export_beacons)
        self.ui.start_frequency_changed.connect(self._start_frequency_changed)
        self.ui.stop_frequency_changed.connect(self._stop_frequency_changed)
        self.ui.gain_changed.connect(self._gain_changed)
        self.ui.sample_rate_changed.connect(self._sample_rate_changed)
        self.ui.smoothness_changed.connect(self._smoothness_changed)
        self.ui.threshold_changed.connect(self._threshold_changed)
        self.ui.dynamic_threshold_changed.connect(self._dynamic_threshold_changed)
        self.ui.beacon_bandwidth_changed.connect(self._beacon_bandwidth_changed)

        self.ui.set_configuration(
            start_frequency=self.min_freq,
            stop_frequency=self.max_freq,
            gain=self.gain,
            sample_rate=self.sample_rate,
            smoothness=self.smothnes,
            threshold=self.threshold,
            dynamic_threshold=self.dynamic_thershold,
            beacon_bandwidth=self.beacon_bandwidth,
        )
        self.ui.set_acquisition_active(False)

        if show:
            self.ui.show()

    @pyqtSlot(float, str)
    def _start_frequency_changed(self, value, unit):
        self.min_freq = self._to_hz(value, unit)

    @pyqtSlot(float, str)
    def _stop_frequency_changed(self, value, unit):
        self.max_freq = self._to_hz(value, unit)

    @pyqtSlot(int)
    def _gain_changed(self, value):
        self.gain = value

    @pyqtSlot(float, str)
    def _sample_rate_changed(self, value, unit):
        self.sample_rate = self._to_hz(value, unit)

    @pyqtSlot(int)
    def _smoothness_changed(self, value):
        self.smothnes = value

    @pyqtSlot(float)
    def _threshold_changed(self, value):
        self.threshold = value

    @pyqtSlot(bool)
    def _dynamic_threshold_changed(self, value):
        self.dynamic_thershold = value

    @pyqtSlot(float, str)
    def _beacon_bandwidth_changed(self, value, unit):
        self.beacon_bandwidth = self._to_hz(value, unit)

    @pyqtSlot()
    def start(self):
        if self.worker is None:
            self._start_acquisition(continuous=True)

    @pyqtSlot()
    def single_shot(self):
        if self.worker is None:
            self._start_acquisition(continuous=False)

    def _start_acquisition(self, continuous):
        config = {
            "sample_rate": self.sample_rate,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "smothnes": self.smothnes,
            "gain": self.gain,
            "threshold": self.threshold,
            "beacon_bandwidth": self.beacon_bandwidth,
            "outer_cut": self.outer_cut,
            "dc_spike_cut": self.dc_spike_cut,
            "dynamic_thershold": self.dynamic_thershold,
            "neighbors": self.neighbors,
        }

        self.thread = QThread()
        self.worker = ScannerWorker(config=config, continuous=continuous)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.frame_ready.connect(self.ui.set_frame)
        self.worker.beacons_ready.connect(self.ui.set_beacons)
        self.worker.center_frequency_ready.connect(self.ui.set_center_frequency)
        self.worker.threshold_ready.connect(self.ui.set_threshold)
        self.worker.beacons_ready.connect(self._beacons_received)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.stopped.connect(self._cleanup_worker)
        self.worker.error.connect(self._acquisition_error)

        self.ui.set_acquisition_active(True)
        self.thread.start()

    @pyqtSlot()
    def stop(self):
        if self.worker is not None:
            self.worker.request_stop()

    @pyqtSlot(str)
    def _acquisition_error(self, message):
        print(f"Scanner error: {message}")

    @pyqtSlot()
    def _cleanup_worker(self):
        thread = self.thread
        self.worker = None
        self.thread = None
        self.ui.set_acquisition_active(False)

        if thread is not None:
            thread.quit()
            if not thread.wait(5000):
                print("Warning: scanner thread did not stop cleanly.")
            thread.deleteLater()

    @staticmethod
    def _to_hz(value, unit):
        multipliers = {
            "Hz": 1.0,
            "kHz": 1e3,
            "MHz": 1e6,
            "GHz": 1e9,
        }
        return float(value) * multipliers[unit]

    def close(self):
        if self.worker is not None:
            self.worker.request_stop()
        if self.thread is not None:
            self.thread.quit()
            if not self.thread.wait(5000):
                print("Warning: scanner thread did not stop cleanly.")

        self.worker = None
        self.thread = None
        self.ui.close()

    @pyqtSlot(object)
    def _beacons_received(self, beacons):
        self.last_beacons = list(beacons)
        try:
            changes = self.channels.report_beacons(self.last_beacons)
            if changes["new"]:
                print(f"New channels detected: {changes['new']}")
            if changes["updated"]:
                print(f"Existing channels updated: {changes['updated']}")
        except Exception as exc:
            print(f"Could not update channels: {exc}")

    @pyqtSlot(str)
    def _export_beacons(self, path):
        if not self.last_beacons:
            print("No beacons to export.")
            return

        try:
            with open(path, "w", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(["frequency_hz", "bandwidth_hz"])
                for beacon in self.last_beacons:
                    writer.writerow([
                        beacon.get("frequency"),
                        beacon.get("bandwidth"),
                    ])
        except OSError as exc:
            print(f"Could not export beacons: {exc}")
            return

        print(f"Exported {len(self.last_beacons)} beacons to {path}")
