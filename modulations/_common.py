"""
Shared building blocks for the voice demodulators (wfm.py, nfm.py, am.py).

    LinearResampler   streaming sample-rate conversion (keeps its position
                      between chunks, so chunk boundaries are invisible)
    AudioChain        high-pass -> low-pass -> optional de-emphasis -> 48 kHz
    FmDemodulator     quadrature FM discriminator + AudioChain
    AmDemodulator     envelope detector with slow AGC + AudioChain

A demodulator file only sets numbers (deviation, filters, de-emphasis).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from demod import AUDIO_RATE, Demodulator


class LinearResampler:
    """Streaming linear-interpolation resampler (input must already be band-limited)."""

    def __init__(self, in_rate, out_rate):
        self._step = float(in_rate) / float(out_rate)    # input samples per output sample
        self._pos = 0.0                                  # next output position in the current buffer
        self._tail = np.empty(0, dtype=np.float64)       # last input sample of the previous chunk

    def process(self, x):
        buf = np.concatenate((self._tail, np.asarray(x, dtype=np.float64)))
        n = buf.size

        if n < 2:
            self._tail = buf
            return np.empty(0, dtype=np.float32)

        count = int(np.ceil((n - 1 - self._pos) / self._step)) if self._pos < n - 1 else 0
        positions = self._pos + self._step * np.arange(count)

        i0 = positions.astype(np.int64)
        frac = positions - i0
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac

        self._pos = self._pos + self._step * count - (n - 1)   # relative to the new tail
        self._tail = buf[-1:]

        return out.astype(np.float32)


class AudioChain:
    """Demodulated baseband at `in_rate` -> clean audio at AUDIO_RATE."""

    def __init__(self, in_rate, highpass_hz, lowpass_hz, deemphasis_us=0.0):
        nyquist = in_rate / 2.0

        self._hp = None
        if highpass_hz and highpass_hz < nyquist * 0.5:
            self._hp = signal.butter(2, highpass_hz / nyquist, "highpass", output="sos")
            self._hp_zi = np.zeros((self._hp.shape[0], 2))

        cutoff = min(lowpass_hz, 0.45 * AUDIO_RATE, 0.45 * in_rate)
        self._lp = signal.butter(4, cutoff / nyquist, "lowpass", output="sos")
        self._lp_zi = np.zeros((self._lp.shape[0], 2))

        self._de = None
        if deemphasis_us:
            alpha = 1.0 - np.exp(-1.0 / (in_rate * deemphasis_us * 1e-6))
            self._de = ([alpha], [1.0, -(1.0 - alpha)])
            self._de_zi = np.zeros(1)

        self._resampler = LinearResampler(in_rate, AUDIO_RATE)

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)

        if self._de is not None:
            x, self._de_zi = signal.lfilter(*self._de, x, zi=self._de_zi)

        if self._hp is not None:
            x, self._hp_zi = signal.sosfilt(self._hp, x, zi=self._hp_zi)

        x, self._lp_zi = signal.sosfilt(self._lp, x, zi=self._lp_zi)

        return self._resampler.process(x)


class FmDemodulator(Demodulator):
    """
    Quadrature (delay-and-multiply) FM discriminator.

    Subclasses set:
        deviation_hz    peak deviation that maps to audio amplitude 1.0
        highpass_hz     removes DC / sub-audible tones (carrier offset ends up here)
        lowpass_hz      audio bandwidth
        deemphasis_us   0 = none (50 for Europe, 75 for the USA, wideband broadcast)
    """

    deviation_hz = 5e3
    highpass_hz = 300.0
    lowpass_hz = 3400.0
    deemphasis_us = 0.0

    def __init__(self):
        self._rate = None
        self.reset()

    def reset(self):
        self._rate = None            # filters are rebuilt on the next chunk
        self._prev = None
        self._chain = None

    def process(self, iq, sample_rate):
        iq = np.asarray(iq, dtype=np.complex64)

        if iq.size == 0:
            return None

        if sample_rate != self._rate or self._chain is None:
            self._rate = float(sample_rate)
            self._prev = None
            self._chain = AudioChain(
                self._rate, self.highpass_hz, self.lowpass_hz, self.deemphasis_us
            )

        x = iq if self._prev is None else np.concatenate(([self._prev], iq))
        self._prev = iq[-1]

        if x.size < 2:
            return None

        # phase change between neighbouring samples = instantaneous frequency
        baseband = np.angle(x[1:] * np.conj(x[:-1])) * (self._rate / (2.0 * np.pi * self.deviation_hz))

        return self._chain.process(baseband)


class AmDemodulator(Demodulator):
    """
    Envelope detector. A slow AGC (running mean of the envelope) removes the
    carrier and keeps the level steady whatever the signal strength is.

    Subclasses set: highpass_hz, lowpass_hz, agc_seconds
    """

    highpass_hz = 60.0
    lowpass_hz = 5000.0
    agc_seconds = 0.5

    def __init__(self):
        self.reset()

    def reset(self):
        self._rate = None
        self._chain = None
        self._mean_zi = None

    def process(self, iq, sample_rate):
        iq = np.asarray(iq, dtype=np.complex64)

        if iq.size == 0:
            return None

        if sample_rate != self._rate or self._chain is None:
            self._rate = float(sample_rate)
            self._chain = AudioChain(self._rate, self.highpass_hz, self.lowpass_hz)
            self._mean_zi = None

        envelope = np.abs(iq).astype(np.float64)

        alpha = 1.0 - np.exp(-1.0 / (self._rate * self.agc_seconds))
        b, a = [alpha], [1.0, -(1.0 - alpha)]

        if self._mean_zi is None:                         # start the mean at the first envelope value
            self._mean_zi = np.array([(1.0 - alpha) * envelope[0]])

        mean, self._mean_zi = _lfilter(b, a, envelope, self._mean_zi)

        baseband = envelope / (mean + 1e-12) - 1.0        # 0 = carrier, +-1 = 100 % modulation

        return self._chain.process(baseband)


def _lfilter(b, a, x, zi):
    return signal.lfilter(b, a, x, zi=zi)
