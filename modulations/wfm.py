"""Wideband FM (broadcast radio): mono audio."""

from modulations._common import FmDemodulator


class WfmDemodulator(FmDemodulator):
    deviation_hz = 75e3       # broadcast FM
    highpass_hz = 30.0
    lowpass_hz = 15e3         # the 19 kHz stereo pilot starts above this
    deemphasis_us = 50.0      # 50 us in Europe and most of the world, 75 us in the USA


DEMODULATOR = WfmDemodulator
