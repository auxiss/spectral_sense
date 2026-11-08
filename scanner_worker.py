from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from scanner import scanner


class ScannerWorker(QObject):
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
            if self.stop_requested:
                self.scanner.running = False

            if self.continuous:
                while not self.stop_requested:
                    self.scanner.scan(
                        center_frequency_callback=self._center_frequency_changed
                    )
                    if self.stop_requested:
                        self._emit_threshold()
                        break
                    self._emit_current_result()
            else:
                self.scanner.scan(
                    center_frequency_callback=self._center_frequency_changed
                )
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

        self.beacons_ready.emit(self.scanner.get_beacons())
        self._emit_threshold()

    def _emit_threshold(self):
        if self.scanner is not None:
            self.threshold_ready.emit(float(self.scanner.get_threshold()))

    def _center_frequency_changed(self, center_frequency):
        self.center_frequency_ready.emit(float(center_frequency))

    @pyqtSlot()
    def request_stop(self):
        self.stop_requested = True
        if self.scanner is not None:
            self.scanner.running = False

    def _close_scanner(self):
        if self.scanner is not None:
            self.scanner.close()
            self.scanner = None
