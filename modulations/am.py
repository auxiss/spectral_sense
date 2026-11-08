"""AM voice (airband, shortwave, CB): envelope detector with AGC."""

from modulations._common import AmDemodulator


class AmVoiceDemodulator(AmDemodulator):
    highpass_hz = 60.0
    lowpass_hz = 5000.0
    agc_seconds = 0.5


DEMODULATOR = AmVoiceDemodulator
