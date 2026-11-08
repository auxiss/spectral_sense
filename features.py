"""
features.py

Everything the observer displays that has to be calculated:

    - channel power spectrum (averaged)
    - channel waterfall (with colour levels that follow the window)
    - IQ density (constellation) plot
    - SNR estimate
    - wideband spectrum

Gain, channel bandwidth and capture sample rate are NOT calculated here.
They are stored with the capture and only displayed.

The same code serves live mode and disk mode:

    live:  window = ChannelWindow(...); window.push(batch); window.compute()
    disk:  features_from_samples(iq, start, sample_rate, center_freq)

Window geometry (kept constant so the ML input is always consistent):

    1024-sample batches x 64 = 65536 samples
    256-point FFT -> 256 rows of 256 bins = 256 x 256 waterfall

Row order: row 0 is the NEWEST row (same as the scanner waterfall).
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ======================================================================
# Display / calculation constants (edit here)
# ======================================================================

BATCH_SIZE = 1024                    # samples read from the SDR at once
CHANNEL_FFT_SIZE = 256               # bins of the channel spectrum / waterfall
WATERFALL_ROWS = 256                 # rows of the channel waterfall
WINDOW_SAMPLES = CHANNEL_FFT_SIZE * WATERFALL_ROWS      # 65536

WIDEBAND_FFT_SIZE = 1024             # bins of the wideband spectrum

IQ_DENSITY_BINS = 128                # the density image is BINS x BINS
IQ_DENSITY_EXTENT = 3.0              # shown range is +/- this many RMS units

# Waterfall colour levels follow the current window.
# 0 / 100 = true minimum / maximum. Raise the low and lower the high
# (for example 1 / 99.9) if a single spike should not stretch the colours.
WATERFALL_LOW_PERCENTILE = 0.0
WATERFALL_HIGH_PERCENTILE = 100.0
WATERFALL_MIN_SPAN_DB = 1.0          # never let the colour range collapse

# SNR estimate
SNR_NOISE_FRACTION = 0.25            # quietest fraction of bins = noise floor
SNR_SIGNAL_MARGIN_DB = 6.0           # bins this far above the floor = signal

EPS = 1e-12                          # avoids log10(0)


# ======================================================================
# Basic building blocks
# ======================================================================

_window_cache: dict[int, np.ndarray] = {}


def _hann(size: int) -> np.ndarray:
    if size not in _window_cache:
        _window_cache[size] = np.hanning(size).astype(np.float32)
    return _window_cache[size]


def power_rows(iq, fft_size: int = CHANNEL_FFT_SIZE) -> np.ndarray:
    """
    Split iq into consecutive non-overlapping blocks of fft_size samples
    and return the LINEAR power spectrum of each block, centred
    (negative frequencies first), shape (rows, fft_size).

    Leftover samples that do not fill a block are ignored.
    """

    iq = np.asarray(iq, dtype=np.complex64)
    rows = iq.size // fft_size

    if rows == 0:
        return np.empty((0, fft_size), dtype=np.float32)

    window = _hann(fft_size)
    blocks = iq[: rows * fft_size].reshape(rows, fft_size) * window

    spectra = np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)
    power = (np.abs(spectra) ** 2) / float(np.sum(window ** 2))

    return power.astype(np.float32)


def to_db(linear_power) -> np.ndarray:
    return (10.0 * np.log10(np.asarray(linear_power) + EPS)).astype(np.float32)


def frequency_axis(center_freq: float, sample_rate: float, fft_size: int) -> np.ndarray:
    """Absolute frequency (Hz) of every FFT bin, matching power_rows()."""

    offsets = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / float(sample_rate)))
    return float(center_freq) + offsets


# ======================================================================
# Individual features
# ======================================================================

def estimate_snr_db(spectrum_db) -> float:
    """
    Rough SNR of the channel from its averaged spectrum (dB).

    Noise floor  = median of the quietest SNR_NOISE_FRACTION of the bins.
    Signal bins  = bins more than SNR_SIGNAL_MARGIN_DB above the floor.
    SNR          = (mean signal-bin power - floor) / floor, in dB.

    Returns 0.0 when no bin rises above the floor.
    """

    spectrum_db = np.asarray(spectrum_db, dtype=np.float64)

    if spectrum_db.size == 0:
        return 0.0

    quiet_count = max(1, int(spectrum_db.size * SNR_NOISE_FRACTION))
    noise_db = float(np.median(np.sort(spectrum_db)[:quiet_count]))

    signal_bins = spectrum_db > noise_db + SNR_SIGNAL_MARGIN_DB

    if not np.any(signal_bins):
        return 0.0

    noise_linear = 10.0 ** (noise_db / 10.0)
    signal_linear = float(np.mean(10.0 ** (spectrum_db[signal_bins] / 10.0)))

    excess = max(signal_linear - noise_linear, EPS)

    return float(10.0 * np.log10(excess / noise_linear))


def waterfall_levels(waterfall_db) -> tuple[float, float]:
    """
    Colour-ramp limits that follow the loudest and quietest parts of
    the current window.
    """

    waterfall_db = np.asarray(waterfall_db)

    if waterfall_db.size == 0:
        return -100.0, -100.0 + WATERFALL_MIN_SPAN_DB

    low = float(np.percentile(waterfall_db, WATERFALL_LOW_PERCENTILE))
    high = float(np.percentile(waterfall_db, WATERFALL_HIGH_PERCENTILE))

    if high - low < WATERFALL_MIN_SPAN_DB:
        high = low + WATERFALL_MIN_SPAN_DB

    return low, high


def iq_density(iq, bins: int = IQ_DENSITY_BINS) -> np.ndarray:
    """
    2-D histogram of the IQ samples, scaled 0..1 (log compressed).

    The samples are first divided by their RMS magnitude, so the picture
    looks the same whatever the gain is. The image is returned with rows
    = Q and columns = I (ready for a row-major image), covering
    +/- IQ_DENSITY_EXTENT RMS units on both axes.
    """

    iq = np.asarray(iq, dtype=np.complex64)

    if iq.size == 0:
        return np.zeros((bins, bins), dtype=np.float32)

    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))

    if rms < EPS:
        return np.zeros((bins, bins), dtype=np.float32)

    scaled = iq / rms
    extent = IQ_DENSITY_EXTENT

    histogram, _, _ = np.histogram2d(
        scaled.real,
        scaled.imag,
        bins=bins,
        range=[[-extent, extent], [-extent, extent]],
    )

    histogram = np.log1p(histogram)
    peak = histogram.max()

    if peak > 0:
        histogram /= peak

    return histogram.T.astype(np.float32)


def wideband_spectrum(iq, sample_rate: float, center_freq: float,
                      fft_size: int = WIDEBAND_FFT_SIZE):
    """
    Averaged spectrum of raw SDR samples.
    Returns (frequencies_hz, power_db), or (empty, empty) if iq is too short.
    """

    rows = power_rows(iq, fft_size)

    if rows.shape[0] == 0:
        return np.empty(0), np.empty(0, dtype=np.float32)

    return (
        frequency_axis(center_freq, sample_rate, fft_size),
        to_db(rows.mean(axis=0)),
    )


# ======================================================================
# Rolling window (live mode)
# ======================================================================

class ChannelWindow:
    """
    Keeps the latest WINDOW_SAMPLES samples of the channel and the
    spectrum rows built from them.

    Each push() only computes the rows for the NEW samples. A 1024-sample
    batch adds 4 rows (1024 / 256) and drops the 4 oldest ones, so the
    waterfall scrolls 4 rows at a time and the averaged spectrum is
    updated with a running sum instead of being recomputed.
    """

    def __init__(self, sample_rate: float, center_freq: float):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)

        self._rows = deque(maxlen=WATERFALL_ROWS)           # linear power, oldest first
        self._row_sum = np.zeros(CHANNEL_FFT_SIZE, dtype=np.float64)

        self._pending = np.empty(0, dtype=np.complex64)     # samples not yet in a row
        self._iq = np.empty(0, dtype=np.complex64)          # latest WINDOW_SAMPLES

    def reset(self):
        self._rows.clear()
        self._row_sum[:] = 0.0
        self._pending = np.empty(0, dtype=np.complex64)
        self._iq = np.empty(0, dtype=np.complex64)

    def push(self, iq_batch):
        iq_batch = np.asarray(iq_batch, dtype=np.complex64)

        if iq_batch.size == 0:
            return

        # Keep the latest WINDOW_SAMPLES for the IQ density plot.
        self._iq = np.concatenate((self._iq, iq_batch))[-WINDOW_SAMPLES:]

        # Turn complete 256-sample blocks into new rows.
        data = np.concatenate((self._pending, iq_batch))
        usable = (data.size // CHANNEL_FFT_SIZE) * CHANNEL_FFT_SIZE
        self._pending = data[usable:]

        if usable == 0:
            return

        for row in power_rows(data[:usable]):
            if len(self._rows) == self._rows.maxlen:
                self._row_sum -= self._rows[0]      # oldest row is dropped by append
            self._rows.append(row)
            self._row_sum += row

        np.maximum(self._row_sum, 0.0, out=self._row_sum)   # guard against rounding

    def row_count(self) -> int:
        return len(self._rows)

    def compute(self) -> dict | None:
        """
        Everything the plots need, or None if no row exists yet.

            frequencies   absolute frequency of each bin (Hz)
            spectrum_db   averaged channel power spectrum (dB)
            waterfall_db  (rows, bins) in dB, row 0 = newest
            levels        (low, high) dB for the waterfall colours
            snr_db        SNR estimate (dB)
            iq_density    (bins, bins) image, 0..1
            sample_rate   capture sample rate (Hz), as given
            rows          number of valid waterfall rows
        """

        if not self._rows:
            return None

        rows = np.asarray(self._rows, dtype=np.float32)

        spectrum_db = to_db(self._row_sum / len(self._rows))
        waterfall_db = to_db(rows[::-1])                    # newest first

        return {
            "frequencies": frequency_axis(
                self.center_freq, self.sample_rate, CHANNEL_FFT_SIZE
            ),
            "spectrum_db": spectrum_db,
            "waterfall_db": waterfall_db,
            "levels": waterfall_levels(waterfall_db),
            "snr_db": estimate_snr_db(spectrum_db),
            "iq_density": iq_density(self._iq),
            "sample_rate": self.sample_rate,
            "rows": len(self._rows),
        }


# ======================================================================
# Disk mode
# ======================================================================

def features_from_samples(iq, start: int, sample_rate: float,
                          center_freq: float = 0.0) -> dict | None:
    """
    Features of a recording, for a window starting at sample `start`
    (any sample, not just multiples of a chunk). A recording shorter
    than WINDOW_SAMPLES gives a smaller waterfall with fewer rows.

    Returns the same dict as ChannelWindow.compute(), or None if fewer
    than CHANNEL_FFT_SIZE samples are available from `start`.
    """

    iq = np.asarray(iq)
    start = max(0, int(start))

    window = ChannelWindow(sample_rate, center_freq)
    window.push(iq[start: start + WINDOW_SAMPLES])

    return window.compute()