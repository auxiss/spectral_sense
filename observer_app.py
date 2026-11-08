"""
observer_app.py

Controller of the observing mode (the counterpart of ScannerApplication).
It connects ObserverUI, ObserverWorker, ChannelManager and the modulation
registry. The UI never touches the manager or the SDR directly.

Source "live": the worker owns the SDR and feeds the plots.
Source "disk": a saved capture is loaded and shown as a still image; the
               scroll bar moves the 65536-sample window sample by sample.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSlot

import features as feat
from audio_player import AudioPlayer
from audio_window import AudioWindow
from demod import DemodRouter
from observer_worker import ObserverWorker


class ObserverApplication(QObject):
    def __init__(self, ui, manager, registry):
        super().__init__()

        self.ui = ui
        self.manager = manager
        self.registry = registry
        self.player = AudioPlayer()                    # one player for every demodulator
        self.demod = DemodRouter(registry, self.player)
        self.audio_window = None

        # ------------------------------------------------------------
        # Configuration (source of truth; the SDR only exists while running)
        # ------------------------------------------------------------
        self.sample_rate = 2.4e6
        self.gain = 20
        self.threshold_db = 6.0
        self.target_samples = 1_000_000
        self.mode = "manual"
        self.source = "live"

        self.thread = None
        self.worker = None
        self._disk = None                  # loaded recording

        # ------------------------------------------------------------
        # UI -> application
        # ------------------------------------------------------------
        ui.source_changed.connect(self._source_changed)
        ui.mode_changed.connect(self._mode_changed)
        ui.start_requested.connect(self.start)
        ui.stop_requested.connect(self.stop)

        ui.target_samples_changed.connect(self._target_changed)
        ui.gain_changed.connect(self._gain_changed)
        ui.threshold_changed.connect(self._threshold_changed)

        ui.channel_selected.connect(self._channel_selected)
        ui.demod_ids_changed.connect(self._demod_ids_changed)
        ui.record_ids_changed.connect(self._record_ids_changed)

        ui.channel_edit_saved.connect(self._channel_edit_saved)
        ui.channel_delete_requested.connect(self._channel_delete_requested)
        ui.modulation_text_changed.connect(self._modulation_text_changed)

        ui.capture_selected.connect(self._capture_selected)
        ui.capture_delete_requested.connect(self._capture_delete_requested)
        ui.disk_position_changed.connect(self._show_disk)

        ui.audio_window_requested.connect(self._show_audio_window)
        # ui.text_window_requested: placeholder, decoded text comes later

        # ------------------------------------------------------------
        # Initial state
        # ------------------------------------------------------------
        ui.set_configuration(
            gain=self.gain,
            threshold=self.threshold_db,
            target_samples=self.target_samples,
            mode=self.mode,
            source=self.source,
        )
        self.refresh_channels()

    # ==================================================================
    # Public
    # ==================================================================

    def is_running(self) -> bool:
        return self.worker is not None

    def refresh_channels(self):
        """Reload the channel list from the manager (also called when the mode is switched to)."""

        channels = self.manager.get_channels()
        decodable = {
            c.id for c in channels
            if self.registry.can_decode(c.modulation, c.protocol)
        }

        self.ui.set_channels(channels, decodable)
        self.ui.set_modulation_choices(self.registry.names())
        self._refresh_captures()

    def close(self):
        self.demod.close()

        if self.audio_window is not None:
            self.audio_window.close()

        if self.worker is not None:
            self.worker.request_stop()

        if self.thread is not None:
            self.thread.quit()

            if not self.thread.wait(5000):
                print("Warning: observer thread did not stop cleanly.")

        self.worker = None
        self.thread = None

    # ==================================================================
    # Acquisition (live)
    # ==================================================================

    @pyqtSlot()
    def start(self):
        if self.worker is not None or self.source != "live":
            return

        record_ids = self.ui.record_ids()
        demod_ids = self.ui.demod_ids()
        channel_id = self.ui.selected_channel_id()

        if self.mode == "manual":
            if channel_id not in self.manager.channels:
                channels = self.manager.get_channels()
                if not channels:
                    print("No channels to observe.")
                    return
                channel_id = channels[0].id
                self.ui.select_channel(channel_id)
                self._refresh_captures()

        elif not record_ids and not demod_ids:
            print("Auto mode needs at least one channel marked for record or demod.")
            return

        config = {
            "sample_rate": self.sample_rate,
            "gain": self.gain,
            "threshold_db": self.threshold_db,
            "mode": self.mode,
            "target_samples": self.target_samples,
            "channel_id": channel_id,
            "record_ids": record_ids,
            "demod_ids": demod_ids,
        }

        self.thread = QThread()
        self.worker = ObserverWorker(self.manager, config, demod_sink=self.demod)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.wideband_ready.connect(self.ui.set_wideband)
        self.worker.features_ready.connect(self.ui.set_features)
        self.worker.channel_changed.connect(self.ui.set_current_channel)
        self.worker.channel_geometry.connect(self._channel_geometry)
        self.worker.activity_changed.connect(self.ui.set_channel_active)
        self.worker.capture_saved.connect(self._capture_saved)
        self.worker.progress.connect(self._progress)

        self.worker.finished.connect(self._cleanup_worker)
        self.worker.stopped.connect(self._cleanup_worker)
        self.worker.error.connect(self._error)

        self.ui.clear_plots()
        self.ui.set_running(True)
        self.thread.start()

    @pyqtSlot()
    def stop(self):
        if self.worker is not None:
            self.worker.request_stop()

    @pyqtSlot(str)
    def _error(self, message):
        print(f"Observer error: {message}")

    @pyqtSlot()
    def _cleanup_worker(self):
        thread = self.thread

        self.worker = None
        self.thread = None

        self.demod.close()                 # stops the sound card, drops what was still queued
        self.ui.set_running(False)

        if thread is not None:
            thread.quit()

            if not thread.wait(5000):
                print("Warning: observer thread did not stop cleanly.")

            thread.deleteLater()

        self.refresh_channels()

    # ==================================================================
    # Worker -> UI
    # ==================================================================

    @pyqtSlot(int, float, float)
    def _channel_geometry(self, _channel_id, start, end):
        self.ui.set_channel_marker(start, end)

    @pyqtSlot(int, str)
    def _capture_saved(self, channel_id, _path):
        if channel_id == self.ui.selected_channel_id():
            self._refresh_captures()

    @pyqtSlot(object)
    def _progress(self, collected):
        self.ui.set_progress(collected, self.target_samples)

    # ==================================================================
    # Audio window
    # ==================================================================

    @pyqtSlot()
    def _show_audio_window(self):
        if self.audio_window is None:
            self.audio_window = AudioWindow(self.player, parent=self.ui)

        self.audio_window.show()
        self.audio_window.raise_()
        self.audio_window.activateWindow()

    # ==================================================================
    # UI -> application: settings
    # ==================================================================

    @pyqtSlot(str)
    def _source_changed(self, source):
        self.source = source
        self._disk = None
        self.ui.clear_plots()
        self._refresh_captures()

    @pyqtSlot(str)
    def _mode_changed(self, mode):
        self.mode = mode

    @pyqtSlot(int)
    def _target_changed(self, value):
        self.target_samples = int(value)

    @pyqtSlot(int)
    def _gain_changed(self, value):
        self.gain = int(value)

        if self.worker is not None:
            self.worker.set_gain(self.gain)

    @pyqtSlot(float)
    def _threshold_changed(self, value):
        self.threshold_db = float(value)

        if self.worker is not None:
            self.worker.set_threshold(self.threshold_db)

    # ==================================================================
    # UI -> application: channels
    # ==================================================================

    @pyqtSlot(int)
    def _channel_selected(self, channel_id):
        if self.worker is not None and self.mode == "manual":
            self.worker.set_manual_channel(channel_id)

        self._disk = None

        if self.source == "disk":
            self.ui.clear_plots()

        self._refresh_captures()

    @pyqtSlot(object)
    def _demod_ids_changed(self, ids):
        if self.worker is not None:
            self.worker.set_demod_ids(ids)

    @pyqtSlot(object)
    def _record_ids_changed(self, ids):
        if self.worker is not None:
            self.worker.set_record_ids(ids)

    @pyqtSlot(str)
    def _modulation_text_changed(self, modulation):
        self.ui.set_protocol_choices(self.registry.protocols(modulation))

    @pyqtSlot(int, object)
    def _channel_edit_saved(self, channel_id, data):
        try:
            if data["start"] is not None:
                # Range edited by the user: trusted overwrite (may merge with an overlapping channel).
                channel = self.manager.refine_channel(channel_id, data["start"], data["end"])
            else:
                channel = self.manager.channels[channel_id]       # range untouched: keep the worker's

            modulation = data["modulation"] or "unknown"
            protocol = data["protocol"] or "unknown"

            self.manager.edit_channel(
                channel.id,
                name=data["name"] or None,          # blank = keep the current name
                modulation=modulation,
                protocol=protocol,
                info=data["info"],
            )

            # New labels typed into the boxes become registry entries.
            self.registry.add_modulation(modulation)
            self.registry.add_protocol(modulation, protocol)

        except Exception as exc:
            print(f"Could not save channel: {exc}")

        self.refresh_channels()

    @pyqtSlot(int)
    def _channel_delete_requested(self, channel_id):
        # Recordings stay on disk (ChannelManager default).
        self.manager.delete_channel(channel_id)
        self._disk = None
        self.refresh_channels()

    # ==================================================================
    # Recordings (disk mode)
    # ==================================================================

    def _refresh_captures(self):
        channel_id = self.ui.selected_channel_id()

        if channel_id in self.manager.channels:
            self.ui.set_captures(self.manager.get_captures(channel_id))
        else:
            self.ui.set_captures([])

    @pyqtSlot(int, str)
    def _capture_selected(self, channel_id, capture_file):
        if self.source != "disk":
            return

        channel = self.manager.channels.get(channel_id)
        row = next(
            (r for r in self.manager.get_captures(channel_id)
             if r["capture_file"] == capture_file),
            None,
        )

        if channel is None or row is None:
            return

        try:
            from scipy.io import wavfile

            rate, data = wavfile.read(str(row["path"]))
            data = np.asarray(data, dtype=np.float32)

            iq = (data[:, 0] + 1j * data[:, 1]) if data.ndim == 2 else data.astype(np.complex64)

        except Exception as exc:
            print(f"Could not load {capture_file}: {exc}")
            return

        self._disk = SimpleNamespace(
            iq=iq.astype(np.complex64),
            rate=float(rate),
            center=(channel.start_freq + channel.end_freq) / 2.0,
            bandwidth=channel.end_freq - channel.start_freq,
            gain=self._to_float(row.get("gain")),
        )

        self.ui.set_disk_recording(iq.size)
        self._show_disk(0)

    @pyqtSlot(int)
    def _show_disk(self, start):
        disk = self._disk

        if disk is None:
            return

        data = feat.features_from_samples(disk.iq, start, disk.rate, 0.0)   # 0 = centred on the channel

        if data is None:
            self.ui.clear_channel_plots()
            return

        data["gain_db"] = disk.gain
        data["bandwidth"] = disk.bandwidth
        self.ui.set_features(data)

    @pyqtSlot(int, str)
    def _capture_delete_requested(self, channel_id, capture_file):
        self._disk = None
        self.ui.clear_plots()

        try:
            self.manager.delete_capture(channel_id, capture_file)
        except Exception as exc:
            print(f"Could not delete {capture_file}: {exc}")

        self._refresh_captures()

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
