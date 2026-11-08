"""
observer_worker.py

Runs the SDR for the observing mode. Lives on a QThread, exactly like
ScannerWorker: create it, moveToThread(), connect thread.started to run().

Two modes (config["mode"]):

    manual  stays on one channel (set_manual_channel) and only moves when told to
    auto    hops through the selected channels; channels marked for demod are
            checked before channels that are only recorded, and the worker
            stays on a channel for as long as it transmits

Per channel visit:

    tune (channel placed away from the SDR's DC spike)
      -> wait for activity (in-band power vs the noise next to the channel)
      -> isolate the channel (channel_isolator) using the scanner's stored
         centre frequency and bandwidth
      -> one burst = one capture_N.wav via ChannelManager.append_capture()

Samples are the unit of collection: in auto mode a channel is finished once
target_samples have been SAVED during this run (earlier sessions do not count).

THREADING: the worker calls the ChannelManager from its own thread while the
GUI may edit it. Give ChannelManager an RLock (see the note in the reply).
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import features as feat
from channel_isolator import ChannelConfig, ChannelIsolator


# ======================================================================
# Constants (edit here)
# ======================================================================

# --- streaming ---------------------------------------------------------
STREAM_CHUNK = 131072              # samples per read (multiple of 256), ~55 ms at 2.4 MS/s
QUEUE_CHUNKS = 16                  # buffered chunks; overflow = gap in the stream
SETTLE_SAMPLES = 16384             # thrown away after every retune

# --- tuning ------------------------------------------------------------
SDR_OFFSET_MIN_HZ = 250e3          # channel sits at least this far from the SDR centre
DC_GUARD_HZ = 10e3                 # keep the DC spike out of the channel (scanner dc_spike_cut)
OUTER_CUT_HZ = 700e3               # unusable band edges (scanner outer_cut)

# --- activity detection ------------------------------------------------
ACTIVITY_FFT = 2048
ACTIVITY_AVERAGES = 8
HANG_TIME_S = 0.4                  # quiet this long = the burst has ended

# --- recording ---------------------------------------------------------
OUTPUT_RATE_FACTOR = 1.5           # capture rate >= this x channel bandwidth
MIN_BURST_SAMPLES = 1024           # shorter bursts are discarded
MAX_BURST_SAMPLES = 5_000_000      # longer bursts continue in a new file (RAM limit)

# --- display -----------------------------------------------------------
FEATURE_INTERVAL_S = 0.1
WIDEBAND_INTERVAL_S = 0.2
WIDEBAND_SAMPLES = 16384


# ======================================================================
# Pure helpers (no Qt, no hardware)
# ======================================================================

def _nice_decimation(d: int) -> bool:
    """True if d only has the prime factors 2, 3, 5, 7 (cheap filter cascade)."""
    for p in (2, 3, 5, 7):
        while d % p == 0:
            d //= p
    return d == 1


def choose_output_rate(sample_rate: float, bandwidth: float) -> float:
    """Largest integer decimation that keeps output rate >= OUTPUT_RATE_FACTOR x bandwidth."""
    d = max(1, int(sample_rate // (bandwidth * OUTPUT_RATE_FACTOR)))
    while d > 1 and not _nice_decimation(d):
        d -= 1
    return sample_rate / d


def band_levels_db(iq, sample_rate, offset_hz, bandwidth_hz):
    """
    (in-band power, noise next to the channel) in dB, or None.
    offset_hz is the channel centre relative to the SDR centre.
    """

    if iq.size < ACTIVITY_FFT:
        return None

    # Average blocks spread over the WHOLE chunk, so a short burst
    # anywhere in the chunk is seen.
    starts = np.linspace(0, iq.size - ACTIVITY_FFT, ACTIVITY_AVERAGES).astype(int)
    blocks = np.concatenate([iq[i:i + ACTIVITY_FFT] for i in starts])
    power = feat.power_rows(blocks, ACTIVITY_FFT).mean(axis=0)

    freqs = np.fft.fftshift(np.fft.fftfreq(ACTIVITY_FFT, 1.0 / sample_rate))
    distance = np.abs(freqs - offset_hz)

    inband = distance <= bandwidth_hz / 2.0
    if not inband.any():
        inband = distance == distance.min()

    ring = (
        (distance > bandwidth_hz * 0.75)
        & (distance <= bandwidth_hz * 2.5)
        & (np.abs(freqs) > DC_GUARD_HZ)
    )
    if not ring.any():
        ring = ~inband

    signal = float(np.mean(power[inband]))
    noise = float(np.median(power[ring]))

    return (
        float(10.0 * np.log10(signal + feat.EPS)),
        float(10.0 * np.log10(noise + feat.EPS)),
    )


def band_snr_db(iq, sample_rate, offset_hz, bandwidth_hz):
    """In-band power minus the noise next to the channel, in dB."""

    levels = band_levels_db(iq, sample_rate, offset_hz, bandwidth_hz)
    return None if levels is None else levels[0] - levels[1]


# ======================================================================
# Reader thread
# ======================================================================

class _Reader(threading.Thread):
    """
    Reads the SDR continuously into a queue, so processing time never
    makes the radio miss samples. If the queue ever overflows, the lost
    chunks are counted in `overflows` (the worker closes the burst).
    """

    def __init__(self, sdr, chunk_size, skip_samples, max_chunks):
        super().__init__(daemon=True)
        self.sdr = sdr
        self.chunk_size = chunk_size
        self.skip_samples = skip_samples
        self.queue = queue.Queue(maxsize=max_chunks)
        self.overflows = 0
        self.error = None
        self._halt = threading.Event()

    def run(self):
        try:
            if self.skip_samples:
                self.sdr.read_samples(self.skip_samples)     # stale data after retune

            while not self._halt.is_set():
                data = np.asarray(
                    self.sdr.read_samples(self.chunk_size), dtype=np.complex64
                )
                try:
                    self.queue.put_nowait(data)
                except queue.Full:
                    self.overflows += 1

        except Exception as exc:
            self.error = exc

    def stop(self):
        self._halt.set()
        self.join(timeout=5.0)


# ======================================================================
# Worker
# ======================================================================

class ObserverWorker(QObject):
    # ---- signals: worker -> UI ----------------------------------------
    wideband_ready = pyqtSignal(object, object)      # frequencies_hz, power_db
    features_ready = pyqtSignal(object)              # features.ChannelWindow.compute() dict
                                                     # + gain_db, bandwidth, center, active
    channel_changed = pyqtSignal(int)                # channel now being observed
    channel_geometry = pyqtSignal(int, float, float) # id, start_hz, end_hz (marker on wideband plot)
    activity_changed = pyqtSignal(int, bool)         # id, transmitting
    capture_saved = pyqtSignal(int, str)             # id, wav path
    progress = pyqtSignal(object)                    # {channel_id: samples saved this run}

    finished = pyqtSignal()
    stopped = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, manager, config, sdr_factory=None, demod_sink=None):
        """
        manager      ChannelManager
        config       dict: sample_rate, gain (percent), threshold_db,
                     mode ("auto" / "manual"), target_samples (auto),
                     channel_id (manual), record_ids, demod_ids
        sdr_factory  callable returning an SDR wrapper (default: RTL_sdr_device)
        demod_sink   optional object with process(iq, sample_rate, channel), called
                     with the isolated channel while it transmits
        """
        super().__init__()

        self.manager = manager
        self.sample_rate = float(config["sample_rate"])
        self.mode = config.get("mode", "manual")
        self.target_samples = int(config.get("target_samples", 0))
        self.threshold_db = float(config.get("threshold_db", 6.0))

        self._gain_percent = config.get("gain", 20)
        self._pending_gain = None
        self._gain_db = None

        self._manual_id = config.get("channel_id")
        self._record_ids = frozenset(config.get("record_ids", ()))
        self._demod_ids = frozenset(config.get("demod_ids", ()))

        self._sdr_factory = sdr_factory
        self.demod_sink = demod_sink
        self.sdr = None

        self.stop_requested = False
        self._collected = {}
        self._last_channel = None
        self._t_wide = 0.0

    # ==================================================================
    # Control from the UI (plain method calls, safe while running)
    # ==================================================================

    def set_manual_channel(self, channel_id):
        self._manual_id = channel_id

    def set_record_ids(self, ids):
        self._record_ids = frozenset(ids)

    def set_demod_ids(self, ids):
        self._demod_ids = frozenset(ids)

    def set_threshold(self, threshold_db):
        self.threshold_db = float(threshold_db)

    def set_gain(self, percent):
        self._pending_gain = percent

    def request_stop(self):
        self.stop_requested = True

    # ==================================================================
    # Thread entry
    # ==================================================================

    @pyqtSlot()
    def run(self):
        try:
            self.stop_requested = False

            if self._sdr_factory is None:
                import RTL_sdr_device
                self._sdr_factory = RTL_sdr_device.rtl_sdr_device

            self.sdr = self._sdr_factory()
            self.sdr.set_sample_rate(self.sample_rate)
            self._apply_gain(self._gain_percent)

            if self.mode == "auto":
                self._run_auto()
            else:
                self._run_manual()

        except Exception as exc:
            self.error.emit(str(exc))

        finally:
            self._close_sdr()

            if self.stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit()

    def _close_sdr(self):
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception:
                pass
            self.sdr = None

    def _apply_gain(self, percent):
        self.sdr.set_gain_percent(percent)
        try:
            self._gain_db = self.sdr.info().get("gain")
        except Exception:
            self._gain_db = None

    # ==================================================================
    # Modes
    # ==================================================================

    def _run_manual(self):
        while not self.stop_requested:
            channel = self.manager.channels.get(self._manual_id)

            if channel is None:
                time.sleep(0.1)
                continue

            self._stay(channel, hop=False)

    def _run_auto(self):
        while not self.stop_requested:
            order = self._auto_order()

            if not order:
                if self._auto_finished():
                    return
                time.sleep(0.1)
                continue

            for channel in order:
                if self.stop_requested:
                    return

                was_active = self._stay(channel, hop=True)

                # A recorded channel just finished: go back to the top of the
                # list so demod channels get checked first again.
                if was_active and channel.id not in self._demod_ids:
                    break

            if self._auto_finished():
                return

    def _is_done(self, channel_id):
        return (
            self.target_samples > 0
            and channel_id in self._record_ids
            and self._collected.get(channel_id, 0) >= self.target_samples
        )

    def _auto_order(self):
        channels = self.manager.channels
        demod = sorted(self._demod_ids)
        record = [
            i for i in sorted(self._record_ids)
            if i not in self._demod_ids and not self._is_done(i)
        ]
        return [channels[i] for i in demod + record if i in channels]

    def _auto_finished(self):
        record = [i for i in self._record_ids if i in self.manager.channels]
        return bool(record) and not self._demod_ids and all(
            self._is_done(i) for i in record
        )

    # ==================================================================
    # One visit to one channel
    # ==================================================================

    @staticmethod
    def _geometry(channel):
        center = (channel.start_freq + channel.end_freq) / 2.0
        bandwidth = channel.end_freq - channel.start_freq
        return center, bandwidth

    def _plan(self, center, bandwidth):
        """SDR centre and channel offset that keep the channel off the DC spike and the edges."""

        usable = self.sample_rate / 2.0 - OUTER_CUT_HZ
        offset = max(SDR_OFFSET_MIN_HZ, bandwidth / 2.0 + 2.0 * DC_GUARD_HZ)
        offset = min(offset, usable - bandwidth / 2.0)

        if offset < bandwidth / 2.0 + DC_GUARD_HZ:
            return None                                   # channel too wide for this setup

        return center - offset, offset

    def _start_reader(self):
        reader = _Reader(self.sdr, STREAM_CHUNK, SETTLE_SAMPLES, QUEUE_CHUNKS)
        reader.start()
        return reader

    def _stay(self, channel, hop):
        """
        Observe one channel.

        hop=True   (auto)   leave as soon as the channel is quiet
        hop=False  (manual) stay until stopped or told to switch

        Returns True if the channel was seen transmitting.
        """

        center, bandwidth = self._geometry(channel)
        plan = self._plan(center, bandwidth)

        if plan is None:
            if not hop:
                time.sleep(0.1)
            return False

        sdr_center, offset = plan
        self.sdr.set_center_freq(sdr_center)

        if channel.id != self._last_channel:
            self._last_channel = channel.id
            self.channel_changed.emit(channel.id)
            self.channel_geometry.emit(channel.id, channel.start_freq, channel.end_freq)

        v = SimpleNamespace(
            ch=channel, sdr_center=sdr_center, offset=offset, bw=bandwidth,
            iso=None, window=None, tracking=False,
            burst=[], burst_len=0, active_len=0, inactive_s=0.0, burst_start=None,
            t_feat=0.0, overflows=0, was_active=False, noise_db=None,
        )

        reader = self._start_reader()

        try:
            while not self.stop_requested:
                if not hop and self._manual_id != v.ch.id:
                    break

                if self._pending_gain is not None:
                    self._end_burst(v)
                    reader.stop()
                    self._apply_gain(self._pending_gain)
                    self._pending_gain = None
                    reader = self._start_reader()
                    v.overflows = 0
                    continue

                try:
                    chunk = reader.queue.get(timeout=0.5)
                except queue.Empty:
                    if reader.error is not None:
                        raise reader.error
                    continue

                if reader.overflows != v.overflows:       # samples were lost: close the burst
                    v.overflows = reader.overflows
                    self._end_burst(v)

                if self._process_chunk(chunk, v, hop):
                    break

            if reader.error is not None:
                raise reader.error

        finally:
            reader.stop()
            self._end_burst(v)

        return v.was_active

    def _process_chunk(self, chunk, v, hop):
        """Handle one wideband chunk. Returns True when the visit should end."""

        levels = band_levels_db(chunk, self.sample_rate, v.offset, v.bw)
        snr = None if levels is None else levels[0] - levels[1]
        active = snr is not None and snr >= self.threshold_db

        if levels is not None:                            # smoothed noise floor (for the threshold line)
            v.noise_db = (
                levels[1] if v.noise_db is None else 0.9 * v.noise_db + 0.1 * levels[1]
            )

        self._publish_wideband(chunk, v)

        # --- transmission just started: isolate using the scanner's channel range
        if active and not v.tracking:
            active = self._begin_burst(v)

        if not v.tracking:
            if hop:
                return True                               # quiet channel: next one
            if v.iso is None:
                self._build_isolator(v)                   # preview so the plots still show the channel

        channel_iq = v.iso.process(chunk)
        v.window.push(channel_iq)
        self._publish_features(v)

        if not v.tracking:
            return False

        # --- a transmission is being followed ---------------------------
        duration = chunk.size / self.sample_rate
        v.inactive_s = 0.0 if active else v.inactive_s + duration

        finished_target = False

        if v.ch.id in self._record_ids and not self._is_done(v.ch.id):
            finished_target = self._add_to_burst(v, channel_iq, active)

        if active and self.demod_sink is not None and v.ch.id in self._demod_ids:
            self.demod_sink.process(channel_iq, v.iso.output_sample_rate, v.ch)

        if finished_target:
            self._end_burst(v)
            return True

        if v.inactive_s >= HANG_TIME_S:
            self._end_burst(v)
            return hop

        return False

    # ------------------------------------------------------------------
    # Burst start / end
    # ------------------------------------------------------------------

    def _begin_burst(self, v):
        try:
            self._build_isolator(v)
        except ValueError:
            return False

        v.tracking = True
        v.was_active = True
        v.burst_start = datetime.now(timezone.utc)

        self.activity_changed.emit(v.ch.id, True)
        return True

    def _end_burst(self, v):
        if v.burst_len:
            self._save_burst(v)

        v.burst, v.burst_len, v.active_len, v.inactive_s = [], 0, 0, 0.0

        if v.tracking:
            v.tracking = False
            v.iso = None
            v.window = None

            center, bandwidth = self._geometry(v.ch)
            v.offset = center - v.sdr_center
            v.bw = bandwidth

            self.activity_changed.emit(v.ch.id, False)

    def _add_to_burst(self, v, channel_iq, active):
        """Append to the burst. Returns True when this channel reached its target."""

        v.burst.append(channel_iq)
        v.burst_len += channel_iq.size

        if active:
            v.active_len = v.burst_len

        if self.mode == "auto" and self.target_samples > 0:
            have = self._collected.get(v.ch.id, 0) + v.burst_len

            if have >= self.target_samples:
                excess = have - self.target_samples
                if excess:
                    v.burst[-1] = v.burst[-1][: channel_iq.size - excess]
                    v.burst_len -= excess
                v.active_len = v.burst_len
                return True

        if v.burst_len >= MAX_BURST_SAMPLES:                # RAM guard: continue in a new file
            v.active_len = v.burst_len
            self._save_burst(v)
            v.burst, v.burst_len, v.active_len = [], 0, 0

        return False

    def _save_burst(self, v):
        if not v.burst or v.iso is None:
            return

        samples = np.concatenate(v.burst)[: v.active_len]
        start = v.burst_start
        v.burst_start = datetime.now(timezone.utc)

        if samples.size < MIN_BURST_SAMPLES:
            return

        path = self.manager.append_capture(
            v.ch.id,
            samples,
            v.iso.output_sample_rate,          # capture sample rate, NOT the channel bandwidth
            gain=self._gain_db,
            timestamp=start,
        )

        self._collected[v.ch.id] = self._collected.get(v.ch.id, 0) + samples.size

        self.capture_saved.emit(v.ch.id, str(path))
        self.progress.emit(dict(self._collected))

    # ------------------------------------------------------------------
    # Isolator and display
    # ------------------------------------------------------------------

    def _build_isolator(self, v):
        center = v.sdr_center + v.offset

        v.iso = ChannelIsolator(ChannelConfig(
            input_sample_rate=self.sample_rate,
            capture_center_freq=v.sdr_center,
            channel_center_freq=center,
            channel_bandwidth=v.bw,
            output_sample_rate=choose_output_rate(self.sample_rate, v.bw),
        ))
        # Centre 0: the channel view shows the offset from the channel centre.
        v.window = feat.ChannelWindow(v.iso.output_sample_rate, 0.0)

    def _publish_wideband(self, chunk, v):
        now = time.monotonic()

        if now - self._t_wide < WIDEBAND_INTERVAL_S:
            return

        self._t_wide = now
        freqs, power = feat.wideband_spectrum(
            chunk[:WIDEBAND_SAMPLES], self.sample_rate, v.sdr_center
        )

        if freqs.size:
            self.wideband_ready.emit(freqs, power)

    def _publish_features(self, v):
        now = time.monotonic()

        if now - v.t_feat < FEATURE_INTERVAL_S:
            return

        v.t_feat = now
        data = v.window.compute()

        if data is None:
            return

        data["gain_db"] = self._gain_db
        data["bandwidth"] = v.bw
        data["center"] = v.sdr_center + v.offset
        data["active"] = v.tracking

        if v.noise_db is not None:
            # Noise floor measured next to the channel, moved onto the channel
            # spectrum's scale (its sample rate is lower, so each bin holds less
            # noise power), plus the user's threshold.
            data["threshold_level_db"] = (
                v.noise_db
                + 10.0 * np.log10(v.window.sample_rate / self.sample_rate)
                + self.threshold_db
            )

        self.features_ready.emit(data)