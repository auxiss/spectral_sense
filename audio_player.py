"""
audio_player.py

One shared audio player for every voice demodulator.

    JitterBuffer   plain numpy, no sound card: takes audio in lumps, gives
                   audio out in fixed blocks. This is the part that can be
                   (and is) tested without audio hardware.
    AudioPlayer    JitterBuffer + sound card (sounddevice) + the controls
                   the Audio playback window shows: volume, mute, pre-roll,
                   output device, level, buffer fill, underrun counters.

Why a buffer: the SDR loop delivers audio in lumps at irregular moments,
the sound card asks for a fixed block at a fixed rate. Playback therefore
only starts once `preroll_s` of audio is queued, and if it ever runs dry it
fades out, waits to refill, and fades back in instead of clicking.

The end of a transmission is NOT an error: when nothing has been pushed for
`end_gap_s`, the rest of the buffer is played out (even if it is shorter than
the pre-roll) and no underrun is counted.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from demod import AUDIO_RATE


class JitterBuffer:
    def __init__(self, audio_rate=AUDIO_RATE, preroll_s=0.4, max_s=2.0,
                 fade_s=0.005, end_gap_s=0.3, clock=time.monotonic):
        self.audio_rate = int(audio_rate)
        self._clock = clock
        self._lock = threading.Lock()

        self._max_n = int(max_s * self.audio_rate)
        self._fade_n = max(1, int(fade_s * self.audio_rate))
        self._end_gap = float(end_gap_s)
        self._preroll_n = 0
        self.set_preroll(preroll_s)

        self._chunks = deque()
        self._offset = 0                    # samples already used from the first chunk
        self._buffered = 0
        self._playing = False
        self._fade_in_left = 0
        self._last_push = None
        self._dry_since = None              # when the buffer ran dry while playing

        self.underruns = 0                  # ran dry in the middle of a transmission
        self.overruns = 0                   # too much queued: oldest audio was dropped

    # ------------------------------------------------------------ settings
    def set_preroll(self, seconds):
        self._preroll_n = max(1, int(float(seconds) * self.audio_rate))
        self._preroll_n = min(self._preroll_n, self._max_n - 1) if self._max_n > 1 else self._preroll_n

    @property
    def preroll_s(self):
        return self._preroll_n / self.audio_rate

    @property
    def max_s(self):
        return self._max_n / self.audio_rate

    @property
    def buffered_s(self):
        with self._lock:
            return self._buffered / self.audio_rate

    @property
    def state(self):  # "idle" / "priming" (waiting for the pre-roll) / "playing"
        with self._lock:
            if self._playing:
                return "playing"
            return "priming" if self._buffered else "idle"

    # --------------------------------------------------------------- input
    def push(self, audio):
        audio = np.asarray(audio, dtype=np.float32).ravel()

        if audio.size == 0:
            return

        with self._lock:
            now = self._clock()

            # The buffer ran dry earlier. If audio comes back soon it was a
            # stall (an underrun); if the silence was long it was simply the
            # end of a transmission and a new one is starting.
            if self._dry_since is not None:
                if now - self._dry_since <= self._end_gap:
                    self.underruns += 1
                self._dry_since = None

            self._chunks.append(audio)
            self._buffered += audio.size
            self._last_push = now

            if self._buffered > self._max_n:             # too much queued: keep only the newest
                self._take(self._buffered - self._preroll_n)
                self.overruns += 1

    def flush(self):
        """Throw everything away (for example after changing the pre-roll)."""
        with self._lock:
            self._chunks.clear()
            self._offset = 0
            self._buffered = 0
            self._playing = False
            self._dry_since = None

    # -------------------------------------------------------------- output
    def read(self, frames):
        """Exactly `frames` samples, zeros when there is nothing to play."""

        with self._lock:
            out = np.zeros(frames, dtype=np.float32)

            idle = 0.0 if self._last_push is None else self._clock() - self._last_push
            stream_ended = idle > self._end_gap

            if not self._playing:
                primed = self._buffered >= self._preroll_n
                drain_tail = self._buffered > 0 and stream_ended

                if not (primed or drain_tail):
                    return out

                self._playing = True
                self._fade_in_left = self._fade_n

            n = min(frames, self._buffered)

            if n:
                out[:n] = self._take(n)

            if self._fade_in_left > 0 and n:
                k = min(self._fade_in_left, n)
                start = self._fade_n - self._fade_in_left
                ramp = (np.arange(start, start + k, dtype=np.float32) + 1.0) / self._fade_n
                out[:k] *= ramp
                self._fade_in_left -= k

            if n < frames:                               # ran dry
                if n:
                    tail = min(n, self._fade_n)
                    out[n - tail:n] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)

                if self._dry_since is None:
                    self._dry_since = self._clock()      # classified when audio resumes (see push)

                self._playing = False

            return out

    # ------------------------------------------------------------ internal
    def _take(self, n):
        """Remove and return the oldest n samples (caller holds the lock)."""

        out = np.empty(n, dtype=np.float32)
        filled = 0

        while filled < n:
            chunk = self._chunks[0]
            k = min(chunk.size - self._offset, n - filled)

            out[filled:filled + k] = chunk[self._offset:self._offset + k]
            filled += k
            self._offset += k

            if self._offset >= chunk.size:
                self._chunks.popleft()
                self._offset = 0

        self._buffered -= n
        return out


class AudioPlayer:
    """JitterBuffer + sound card + the user's controls."""

    BLOCK = 1024

    def __init__(self, audio_rate=AUDIO_RATE, preroll_s=0.4, stream_factory=None):
        self.audio_rate = int(audio_rate)
        self.buffer = JitterBuffer(audio_rate, preroll_s)

        self.volume = 1.0                   # 0.0 .. 2.0
        self.muted = False
        self.device = None                  # None = system default
        self.level = 0.0                    # peak of the latest block, decays
        self.error = None

        self._stream_factory = stream_factory or self._sounddevice_stream
        self._stream = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ controls
    def set_volume(self, volume):
        self.volume = max(0.0, float(volume))

    def set_muted(self, muted):
        self.muted = bool(muted)

    def set_preroll(self, seconds):
        self.buffer.set_preroll(seconds)

    def set_device(self, device):
        self.device = device

        if self._stream is not None:
            self.stop()
            self.start()

    @staticmethod
    def devices():
        """[(index, name)] of the output devices, or [] without sounddevice."""
        try:
            import sounddevice as sd

            return [
                (i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] > 0
            ]
        except Exception:
            return []

    # ------------------------------------------------------------- running
    @property
    def running(self):
        return self._stream is not None

    def start(self):
        with self._lock:
            if self._stream is not None:
                return

            try:
                self._stream = self._stream_factory(self.audio_rate, self._callback, self.device)
                self._stream.start()
                self.error = None
            except Exception as exc:
                self._stream = None
                self.error = str(exc)

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        self.buffer.flush()
        self.level = 0.0

    def push(self, audio):
        """Called by the demodulators (from the worker thread)."""

        if self._stream is None:
            if self.error is not None:
                return                       # no sound card: nothing to play to
            self.start()
            if self._stream is None:
                return

        self.buffer.push(audio)

    # ------------------------------------------------------------ internal
    def _callback(self, outdata, frames, time_info, status):
        data = self.buffer.read(frames)
        gain = 0.0 if self.muted else self.volume
        data = np.clip(data * gain, -1.0, 1.0)

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        self.level = max(peak, self.level * 0.9)

        outdata[:, 0] = data

    def _sounddevice_stream(self, rate, callback, device):
        import sounddevice as sd

        return sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32",
            device=device, blocksize=self.BLOCK, callback=callback,
        )
