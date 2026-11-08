import sys

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication

import csv

from scanner_ui import ScannerUI
from scanner import scanner
import known_ranges
from channel_manager import ChannelManager


class ScannerWorker(QObject):
    # powers, frequencies, max hold, threshold curve
    frame_ready = pyqtSignal(object, object, object, object)
    beacons_ready = pyqtSignal(object)
    center_frequency_ready = pyqtSignal(float)
    threshold_ready = pyqtSignal(float)

    finished = pyqtSignal()
    stopped = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config, continuous=False):
        super().__init__()

        self.config = config
        self.continuous = continuous

        self.scanner = None
        self.stop_requested = False

    @pyqtSlot()
    def run(self):
        try:
            self.stop_requested = False

            # Create a completely fresh scanner for every acquisition.
            self.scanner = scanner(
                sample_rate=self.config["sample_rate"],
                min_freq=self.config["min_freq"],
                max_freq=self.config["max_freq"],
                smothnes=self.config["smothnes"],
                gain=self.config["gain"],
                threshold=self.config["threshold"],
                beacon_bandwidth=self.config["beacon_bandwidth"],
                outer_cut=self.config["outer_cut"],
                dc_spike_cut=self.config["dc_spike_cut"],
                dynamic_thershold=self.config["dynamic_thershold"],
                neighbors=self.config["neighbors"],
            )

            if self.continuous:
                # Continuous mode:
                # keep running complete scans until STOP is requested.
                while not self.stop_requested:

                    self.scanner.scan(center_frequency_callback=self._center_frequency_changed)

                    if self.stop_requested:
                        self._emit_threshold()
                        break

                    self._emit_current_result()

            else:
                # Single Shot:
                # perform exactly one complete scan.
                self.scanner.scan(center_frequency_callback=self._center_frequency_changed)

                if not self.stop_requested:
                    self._emit_current_result()
                else:
                    self._emit_threshold()

        except Exception as exc:
            self.error.emit(str(exc))

        finally:
            self._close_scanner()

            if self.stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit()

    def _emit_current_result(self):
        if self.scanner is None:
            return

        frame = self.scanner.get_frame()

        if frame is not None:
            powers = frame.get("power")
            frequencies = frame.get("freqs")
            max_hold = frame.get("max")
            threshold_curve = frame.get("threshold")

            if powers is not None and frequencies is not None and max_hold is not None:
                self.frame_ready.emit(powers, frequencies, max_hold, threshold_curve)

        beacons = self.scanner.get_beacons()
        self.beacons_ready.emit(beacons)
        self._emit_threshold()

    def _emit_threshold(self):
        # get_threshold() is the mean of the threshold curve (a single number),
        # used by the UI to calibrate the waterfall colours.
        if self.scanner is not None:
            self.threshold_ready.emit(
                float(self.scanner.get_threshold())
            )



    def _center_frequency_changed(self, center_frequency):
        try:
            self.center_frequency_ready.emit(float(center_frequency))
        except Exception:
            pass

    def request_stop(self):
        """
        Request that the worker stops.

        scanner.scan() checks scanner.running between sweep steps, so
        setting running=False causes the current scan to terminate.
        """
        self.stop_requested = True

        if self.scanner is not None:
            try:
                self.scanner.running = False
            except Exception:
                pass

    def _close_scanner(self):
        if self.scanner is not None:
            try:
                self.scanner.close()
            except Exception:
                pass

            self.scanner = None


class ScannerApplication(QObject):
    def __init__(self):
        super().__init__()

        
        self.ui = ScannerUI()

        self.last_beacons = []

        ranges_manager = known_ranges.ranges_manager("no_file")
        ranges = ranges_manager.get_ranges()
        self.ui.set_frequency_ranges(ranges)
        self.channels = ChannelManager("channels")

        # ------------------------------------------------------------
        # Application configuration
        #
        # These are the source of truth.
        # The scanner itself does not exist while the application is idle.
        # ------------------------------------------------------------

        self.sample_rate = 2.4e6
        self.min_freq = 60e6
        self.max_freq = 150e6

        self.smothnes = 100
        self.gain = 20
        self.threshold = 2

        self.beacon_bandwidth = 100e3
        self.dynamic_thershold = True

        # Scanner parameters not currently exposed in the UI.
        self.outer_cut = 700e3
        self.dc_spike_cut = 10e3
        self.neighbors = 100        # bins on each side for the local threshold mean

        # ------------------------------------------------------------
        # Acquisition state
        # ------------------------------------------------------------

        self.thread = None
        self.worker = None

        # ------------------------------------------------------------
        # Connect UI signals
        # ------------------------------------------------------------

        self.ui.start_requested.connect(self.start)
        self.ui.stop_requested.connect(self.stop)
        self.ui.single_shot_requested.connect(self.single_shot)

        self.ui.export_beacons_requested.connect(
            self._export_beacons
        )

        self.ui.start_frequency_changed.connect(
            self._start_frequency_changed
        )

        self.ui.stop_frequency_changed.connect(
            self._stop_frequency_changed
        )

        self.ui.gain_changed.connect(
            self._gain_changed
        )

        self.ui.sample_rate_changed.connect(
            self._sample_rate_changed
        )

        self.ui.smoothness_changed.connect(
            self._smoothness_changed
        )

        self.ui.threshold_changed.connect(
            self._threshold_changed
        )

        self.ui.dynamic_threshold_changed.connect(
            self._dynamic_threshold_changed
        )

        self.ui.beacon_bandwidth_changed.connect(
            self._beacon_bandwidth_changed
        )

        # ------------------------------------------------------------
        # Load initial values into the UI
        # ------------------------------------------------------------

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
        self.ui.show()

    # ================================================================
    # UI -> application configuration
    # ================================================================

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

    # ================================================================
    # Acquisition
    # ================================================================

    @pyqtSlot()
    def start(self):
        if self.worker is not None:
            return

        self._start_acquisition(continuous=True)

    @pyqtSlot()
    def single_shot(self):
        if self.worker is not None:
            return

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
        self.worker = ScannerWorker(
            config=config,
            continuous=continuous,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.frame_ready.connect(
            self.ui.set_frame
        )

        self.worker.beacons_ready.connect(
            self.ui.set_beacons
        )

        self.worker.center_frequency_ready.connect(
            self.ui.set_center_frequency
        )

        self.worker.threshold_ready.connect(
            self.ui.set_threshold
        )

        self.worker.beacons_ready.connect(
            self._beacons_received
        )

        self.worker.finished.connect(
            self._acquisition_finished
        )

        self.worker.stopped.connect(
            self._acquisition_stopped
        )

        self.worker.error.connect(
            self._acquisition_error
        )

        self.thread.start()

        self.ui.set_acquisition_active(True)

    @pyqtSlot()
    def stop(self):
        if self.worker is None:
            return

        self.worker.request_stop()

    # ================================================================
    # Worker completion
    # ================================================================

    @pyqtSlot()
    def _acquisition_finished(self):
        self._cleanup_worker()

    @pyqtSlot()
    def _acquisition_stopped(self):
        self._cleanup_worker()

    @pyqtSlot(str)
    def _acquisition_error(self, message):
        print(f"Scanner error: {message}")
        self._cleanup_worker()

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

    # ================================================================
    # Utilities
    # ================================================================

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




    # ================================================================
    # File actions
    # ================================================================
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
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["frequency_hz", "bandwidth_hz"]
                )

                for beacon in self.last_beacons:
                    writer.writerow([
                        beacon.get("frequency"),
                        beacon.get("bandwidth"),
                    ])

            print(f"Exported {len(self.last_beacons)} beacons to {path}")

        except OSError as exc:
            print(f"Could not export beacons: {exc}")


def main():
    app = QApplication(sys.argv)

    application = ScannerApplication()

    app.aboutToQuit.connect(application.close)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()