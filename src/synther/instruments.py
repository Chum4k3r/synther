# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple
import numpy as np

from synther.envelopes import Envelope, EnvelopeADSR
from synther.oscilators import sinewave, warmsaw, noise, squarewave, triangle, sawtooth


Oscilator = Callable[[float, float, np.ndarray], np.ndarray]


class Instrument:
    def __init__(self, amplitude: float, envelope: Envelope) -> None:
        self.harmonics: List[Tuple[int, np.ndarray, float, Oscilator]] = []
        self.envelope = envelope
        self.amplitude = amplitude

    def waveform(self, frequency: float, timestamps: np.ndarray) -> np.ndarray:
        sound = (
            (s := sum(
                    [
                        h_amp * oscilator(h_num*frequency, h_phase, timestamps)
                        for h_num, h_amp, h_phase, oscilator in self.harmonics
                    ]
                )
            ) / np.max(np.abs(s))
        )
        return self.amplitude * sound

    def add_harmonic(self, number: int, oscilator: Oscilator, left_amp: float = 1.0, right_amp: float = 1.0, phase: float = 0.0) -> None:
        h_amp = np.array([[left_amp], [right_amp]])
        self.harmonics.append((number, h_amp, phase, oscilator))


class Harmonica(Instrument):
    """A Harmonica."""

    def __init__(self, amplitude: float = 0.6):
        """Harmonica like sounding instrument."""
        envelope = EnvelopeADSR(attack_dur=0.1,
                                decay_dur=0.05,
                                release_dur=0.1,
                                attack_amp=1.0,
                                sustain_amp=0.8)
        super().__init__(amplitude, envelope)

        self.add_harmonic(1, warmsaw)
        self.add_harmonic(2, squarewave, left_amp=0.5, right_amp=0.2)
        self.add_harmonic(2, squarewave, left_amp=0.2, right_amp=0.5, phase=90)
        self.add_harmonic(3, squarewave, left_amp=0.2, right_amp=0.2)
        self.add_harmonic(4, squarewave, left_amp=0.1, right_amp=0.1)
        self.add_harmonic(0, noise, left_amp=0.05, right_amp=0.05)
        return
