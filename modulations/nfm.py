"""Narrowband FM (two-way radio, walkie-talkies, amateur FM)."""

from modulations._common import FmDemodulator


class NfmDemodulator(FmDemodulator):
    deviation_hz = 5e3        # 25 kHz channels; 12.5 kHz channels use 2.5 kHz (just quieter)
    highpass_hz = 300.0       # also drops sub-audible CTCSS tones
    lowpass_hz = 3400.0       # voice
    deemphasis_us = 0.0


DEMODULATOR = NfmDemodulator
