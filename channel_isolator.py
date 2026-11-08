"""
Streaming channel isolator: wideband IQ in -> single narrowband channel out.

Pipeline (every stage keeps state between chunks, so chunk boundaries are
invisible in the output):

    IQ chunk -> FrequencyTranslator -> FIRDecimator stage(s) -> narrowband IQ

Examples:
    # Offline: pull a 200 kHz channel at 100.3 MHz out of a WAV recording
    python channel_isolator.py --input capture.wav --capture-freq 100.0e6 \
        --channel-freq 100.3e6 --bandwidth 200e3 --out-rate 300e3 \
        --output channel.wav

    # Live from an RTL-SDR (needs `pip install pyrtlsdr`)
    python channel_isolator.py --rtlsdr --rate 2.4e6 --capture-freq 100.0e6 \
        --channel-freq 100.3e6 --bandwidth 200e3 --out-rate 300e3 \
        --output channel.wav
"""

from __future__ import annotations

import argparse
import dataclasses
import wave
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """Describes the wideband capture and the channel you want out of it."""
    input_sample_rate: float      # Hz, sample rate of the wideband capture
    capture_center_freq: float    # Hz, SDR tuner frequency
    channel_center_freq: float    # Hz, absolute frequency of the target channel
    channel_bandwidth: float      # Hz, width of the channel to keep
    output_sample_rate: float     # Hz, desired rate after decimation
    stopband_attenuation_db: float = 60.0
    transition_width_hz: float = 0.0   # 0 -> widest transition that avoids aliasing
    max_stage_decimation: int = 10     # cascade stages decimate by at most this

    @property
    def offset_hz(self) -> float:
        """Channel offset from the capture's center frequency."""
        return self.channel_center_freq - self.capture_center_freq


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
class SampleSource:
    """Abstract source of IQ chunks (SDR hardware, WAV file, socket...)."""

    sample_rate: float = 0.0

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield complex64 arrays, forever or until end of input."""
        raise NotImplementedError


class WavIQSource(SampleSource):
    """Reads a stereo WAV where left = I and right = Q (SDR#/HDSDR style)."""

    def __init__(self, path: str, chunk_size: int = 65536):
        from scipy.io import wavfile
        rate, data = wavfile.read(path, mmap=True)  # mmap: no full load in RAM
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Expected a 2-channel (I/Q) WAV file")
        self.sample_rate = float(rate)
        self._data = data
        self.chunk_size = chunk_size

    def chunks(self) -> Iterator[np.ndarray]:
        for start in range(0, self._data.shape[0], self.chunk_size):
            yield self._to_complex(self._data[start:start + self.chunk_size])

    @staticmethod
    def _to_complex(block: np.ndarray) -> np.ndarray:
        if block.dtype == np.uint8:
            x = (block.astype(np.float32) - 127.5) / 127.5
        elif np.issubdtype(block.dtype, np.integer):
            x = block.astype(np.float32) / float(-np.iinfo(block.dtype).min)
        else:
            x = block.astype(np.float32)
        return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)


class RtlSdrSource(SampleSource):
    """Live samples from an RTL-SDR dongle via pyrtlsdr."""

    def __init__(self, center_freq: float, sample_rate: float,
                 gain="auto", chunk_size: int = 262144):
        self.center_freq = center_freq
        self.sample_rate = float(sample_rate)
        self.gain = gain
        self.chunk_size = chunk_size  # must be a multiple of 256

    def chunks(self) -> Iterator[np.ndarray]:
        from rtlsdr import RtlSdr  # imported lazily so WAV mode needs no driver
        sdr = RtlSdr()
        try:
            sdr.sample_rate = self.sample_rate
            sdr.center_freq = self.center_freq
            sdr.gain = self.gain
            while True:
                # NOTE: synchronous read. If your processing is slower than
                # the hardware, samples get dropped. For heavy demod chains,
                # read in a thread and hand chunks over through a queue.
                yield np.asarray(sdr.read_samples(self.chunk_size),
                                 dtype=np.complex64)
        finally:
            sdr.close()


# --------------------------------------------------------------------------
# DSP blocks
# --------------------------------------------------------------------------
class FrequencyTranslator:
    """Digital local oscillator: shifts the channel down to 0 Hz."""

    def __init__(self, offset_hz: float, sample_rate: float):
        self._fs = float(sample_rate)
        self._phase = 0.0       # radians, carried across chunks
        self._phase_inc = 0.0   # radians per sample
        self.set_offset(offset_hz)

    def set_offset(self, offset_hz: float) -> None:
        """Retune on the fly. Phase is kept, so there is no phase jump."""
        if abs(offset_hz) >= self._fs / 2:
            raise ValueError("Offset is outside the capture's Nyquist range")
        self._phase_inc = -2.0 * np.pi * offset_hz / self._fs

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq)
        # float64 phase math: float32 would drift audibly over long runs
        phases = self._phase + self._phase_inc * np.arange(n, dtype=np.float64)
        lo = np.exp(1j * phases).astype(np.complex64)
        self._phase = (self._phase + self._phase_inc * n) % (2.0 * np.pi)
        return iq * lo

    def reset(self) -> None:
        self._phase = 0.0


class FIRDecimator:
    """Low-pass FIR filter + downsample, with state carried across chunks."""

    def __init__(self, taps: np.ndarray, decimation: int):
        self._taps = np.asarray(taps, dtype=np.float64)
        self._decim = int(decimation)
        self.reset()

    def reset(self) -> None:
        # Filter delay line (the last len(taps)-1 inputs of the previous chunk)
        self._zi = np.zeros(len(self._taps) - 1, dtype=np.complex128)
        # How many input samples to skip before the next one we keep, so the
        # decimation grid stays aligned across chunk boundaries.
        self._skip = 0

    @staticmethod
    def design_taps(sample_rate: float, cutoff_hz: float,
                    transition_hz: float, atten_db: float) -> np.ndarray:
        """Kaiser-window low-pass. Cutoff sits mid-transition."""
        nyq = sample_rate / 2.0
        numtaps, beta = signal.kaiserord(atten_db, transition_hz / nyq)
        numtaps |= 1  # force odd length -> symmetric, integer group delay
        return signal.firwin(numtaps, cutoff_hz, window=("kaiser", beta),
                             fs=sample_rate)

    def process(self, iq: np.ndarray) -> np.ndarray:
        if iq.size == 0:
            return iq[:0].astype(np.complex64)
        filtered, self._zi = signal.lfilter(self._taps, 1.0, iq, zi=self._zi)
        out = filtered[self._skip::self._decim]
        self._skip = (self._skip - len(filtered)) % self._decim
        return out.astype(np.complex64)


def _split_decimation(total: int, max_stage: int) -> List[int]:
    """Factor `total` into a few stage factors, each <= max_stage if possible.

    Smaller factors come first: early stages run at the highest sample rate,
    and a small decimation there allows a wide, cheap transition band.
    """
    if total == 1:
        return [1]
    primes, n, p = [], total, 2
    while p * p <= n:
        while n % p == 0:
            primes.append(p)
            n //= p
        p += 1
    if n > 1:
        primes.append(n)

    stages: List[int] = []
    for prime in sorted(primes, reverse=True):
        for i, s in enumerate(stages):
            if s * prime <= max_stage:
                stages[i] = s * prime
                break
        else:
            stages.append(prime)
    return sorted(stages)


# --------------------------------------------------------------------------
# Top-level block
# --------------------------------------------------------------------------
class ChannelIsolator:
    """Feed it wideband chunks, get narrowband chunks of one channel."""

    def __init__(self, config: ChannelConfig):
        self.config = dataclasses.replace(config)  # private copy
        cfg = self.config

        if cfg.output_sample_rate <= cfg.channel_bandwidth:
            raise ValueError("output_sample_rate must exceed channel_bandwidth")
        ratio = cfg.input_sample_rate / cfg.output_sample_rate
        decim = int(round(ratio))
        if decim < 1 or abs(ratio - decim) > 1e-6 * ratio:
            raise ValueError("input/output sample rate ratio must be an integer")
        self._check_offset(cfg.offset_hz)

        self.decimation = decim
        self.output_sample_rate = cfg.input_sample_rate / decim

        self._translator = FrequencyTranslator(cfg.offset_hz,
                                               cfg.input_sample_rate)
        self._stages: List[FIRDecimator] = []
        fs = cfg.input_sample_rate
        for d in _split_decimation(decim, cfg.max_stage_decimation):
            stage_out = fs / d
            # Stopband starts where aliases would fold into the channel edge:
            # stage_out - bandwidth/2. Passband edge is bandwidth/2.
            transition = stage_out - cfg.channel_bandwidth
            if d == 1:
                # With no rate change, the stopband can start at Nyquist;
                # keep the transition between the channel edge and Nyquist.
                transition /= 2
            if cfg.transition_width_hz > 0:
                transition = min(cfg.transition_width_hz, transition)
            cutoff = cfg.channel_bandwidth / 2 + transition / 2
            taps = FIRDecimator.design_taps(
                fs, cutoff, transition, cfg.stopband_attenuation_db)
            self._stages.append(FIRDecimator(taps, d))
            fs = stage_out

    def _check_offset(self, offset_hz: float) -> None:
        cfg = self.config
        if abs(offset_hz) + cfg.channel_bandwidth / 2 > cfg.input_sample_rate / 2:
            raise ValueError("Channel does not fit inside the captured span")

    def process(self, iq_chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(iq_chunk, dtype=np.complex64)
        x = self._translator.process(x)
        for stage in self._stages:
            x = stage.process(x)
        return x

    def retune(self, new_channel_center_freq: float) -> None:
        """Hop to another channel inside the same capture. Filters untouched."""
        offset = new_channel_center_freq - self.config.capture_center_freq
        self._check_offset(offset)
        self._translator.set_offset(offset)
        self.config.channel_center_freq = new_channel_center_freq

    def reset(self) -> None:
        self._translator.reset()
        for stage in self._stages:
            stage.reset()

    def describe(self) -> str:
        taps = [len(s._taps) for s in self._stages]
        factors = [s._decim for s in self._stages]
        return (f"offset {self.config.offset_hz/1e3:+.1f} kHz, "
                f"decimation {self.decimation} = {factors}, taps {taps}, "
                f"output rate {self.output_sample_rate:.0f} Hz")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
class WavIQSink:
    """Streams complex samples to a stereo int16 WAV (left = I, right = Q)."""

    def __init__(self, path: str, sample_rate: float):
        self._wav = wave.open(path, "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)
        self._wav.setframerate(int(round(sample_rate)))

    def write(self, iq: np.ndarray) -> None:
        pcm = np.empty((len(iq), 2), dtype=np.int16)
        pcm[:, 0] = np.clip(iq.real * 32767, -32768, 32767).astype(np.int16)
        pcm[:, 1] = np.clip(iq.imag * 32767, -32768, 32767).astype(np.int16)
        self._wav.writeframes(pcm.tobytes())

    def close(self) -> None:
        self._wav.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Wideband IQ WAV file")
    src.add_argument("--rtlsdr", action="store_true", help="Use a live RTL-SDR")
    ap.add_argument("--rate", type=float, default=2.4e6,
                    help="RTL-SDR sample rate in Hz (live mode only)")
    ap.add_argument("--gain", default="auto", help="RTL-SDR gain (live mode)")
    ap.add_argument("--capture-freq", type=float, required=True,
                    help="Center frequency of the wideband capture, Hz")
    ap.add_argument("--channel-freq", type=float, required=True,
                    help="Absolute frequency of the channel to extract, Hz")
    ap.add_argument("--bandwidth", type=float, required=True,
                    help="Channel bandwidth, Hz")
    ap.add_argument("--out-rate", type=float, required=True,
                    help="Output sample rate, Hz (input/output must be an integer)")
    ap.add_argument("--output", required=True, help="Output IQ WAV path")
    args = ap.parse_args()

    if args.rtlsdr:
        gain = args.gain if args.gain == "auto" else float(args.gain)
        source: SampleSource = RtlSdrSource(args.capture_freq, args.rate, gain)
    else:
        source = WavIQSource(args.input)

    isolator = ChannelIsolator(ChannelConfig(
        input_sample_rate=source.sample_rate,
        capture_center_freq=args.capture_freq,
        channel_center_freq=args.channel_freq,
        channel_bandwidth=args.bandwidth,
        output_sample_rate=args.out_rate,
    ))
    print(isolator.describe())

    with WavIQSink(args.output, isolator.output_sample_rate) as sink:
        try:
            for wideband_chunk in source.chunks():
                narrowband_chunk = isolator.process(wideband_chunk)
                sink.write(narrowband_chunk)
                # Live demod goes here instead of (or alongside) the sink,
                # e.g. demodulator.process(narrowband_chunk).
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()