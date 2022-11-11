# -*- coding: utf-8 -*-

import numpy as np


def _omega(freq: float) -> float:
    return 2 * np.pi * freq


def sinewave(frequency: float, phase: float, timestamps: np.ndarray,
             LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    radians = _omega(frequency * timestamps + phase) + LFOamp * frequency * np.sin(_omega(LFOfreq) * timestamps)
    out = np.sin(radians)
    return out


def squarewave(frequency: float, phase: float, timestamps: np.ndarray,
             LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    radians = _omega(frequency * timestamps + phase) + LFOamp * frequency * np.sin(_omega(LFOfreq) * timestamps)
    out = np.sin(radians)
    out[:] += 2. * (np.sin(radians) >= 0) - 1.
    return out


def triangle(frequency: float, phase: float, timestamps: np.ndarray,
             LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    radians = _omega(frequency * timestamps + phase) + LFOamp * frequency * np.sin(_omega(LFOfreq) * timestamps)
    out = np.arcsin(np.sin(radians)) * 2. / np.pi
    return out


def warmsaw(frequency: float, phase: float, timestamps: np.ndarray,
             LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    radians = _omega(frequency * timestamps + phase) + LFOamp * frequency * np.sin(_omega(LFOfreq) * timestamps)
    out = np.zeros(timestamps.shape)
    for n in range(1, 30):
        out[:] += (np.sin(n * radians) / n)
    return out * (2 / np.pi)


def sawtooth(frequency: float, phase: float, timestamps: np.ndarray,
             LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    return (2. / np.pi) * (frequency * np.pi * (timestamps % (1/frequency)) - (np.pi/2.))


def noise(frequency: float, phase: float, timestamps: np.ndarray,
          LFOfreq: float = 0., LFOamp: float = 0.) -> np.ndarray:
    out = np.random.standard_normal(timestamps.shape)
    out /= np.max(np.abs(out))
    return out
