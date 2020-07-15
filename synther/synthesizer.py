# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:18:07 2020

@author: joaovitor
"""

import numpy as np
import numba as nb
import multiprocessing as mp
import sounddevice as sd
from synther.notes import get_note_freq


@nb.njit
def omega(freq: float) -> float:
    return 2*np.pi*freq


@nb.njit(parallel=True)
def oscilator(oscType: str, freq: float, timeSpace: np.ndarray) -> np.ndarray:
    if oscType == 'sine':
        return np.sin(omega(freq) * timeSpace)
    elif oscType == 'square':
        return 2. * (np.sin(omega(freq) * timeSpace) >= 0) - 1.
    elif oscType == 'triangle':
        return np.arcsin(np.sin(omega(freq) * timeSpace)) * 2. / np.pi
    elif oscType == 'warmsaw':
        out = np.zeros((timeSpace.shape[0], ))
        for n in nb.prange(1, 30):
            out[:] += (np.sin(n * omega(freq) * timeSpace) / n)
        return out * (2 / np.pi)
    elif oscType == 'sawtooth':
        return (1. / np.pi) * ( freq * np.pi * (timeSpace % (1/freq)) - (np.pi/2.))
    elif oscType == 'noise':
        return np.random.randn(timeSpace.shape[0])
    else:
        return np.zeros((timeSpace.shape[0], ))


_ = oscilator('warmsaw', np.pi, np.linspace(0, 63/44100, 64))
del _


class Envelope:
    def __init__(self, attack: float = 0.01, decay: float = 0.005, release: float = 0.02,
                 start: float = 1.0, sustain: float = 0.8):
        self.attackTime = attack
        self.decayTime = decay
        self.releaseTime = release

        self.startAmp = start
        self.sustainAmp = sustain

        self.triggerOnTime = 0.
        self.triggerOffTime = 0.

        self.noteOn = mp.Event()
        return

    def amplitude(self, timeSpace):
        amplitude = np.zeros(timeSpace.shape[0])
        lifeTime = timeSpace - self.triggerOnTime
        if self.noteOn.is_set():
            # ADS
            atk = lifeTime <= self.attackTime
            dec = lifeTime > self.attackTime
            dec[dec] = lifeTime[dec] <= (self.attackTime + self.decayTime)
            sus = lifeTime > self.attackTime + self.decayTime
            amplitude[atk] = (lifeTime[atk] / self.attackTime) * self.startAmp
            amplitude[dec] = ((lifeTime[dec] - self.attackTime) / self.decayTime) * \
                (self.sustainAmp - self.startAmp) + self.startAmp
            amplitude[sus] = self.sustainAmp
        else:
            # R
            amplitude[:] = ((timeSpace - self.triggerOffTime) / self.releaseTime) * \
                (0 - self.sustainAmp) + self.sustainAmp

        safe = amplitude <= 0.0005
        amplitude[safe] = 0.0

        return amplitude

    def note_on(self, timeOn):
        self.triggerOnTime = timeOn
        self.noteOn.set()
        return

    def note_off(self, timeOff):
        self.triggerOnTime = timeOff
        self.noteOn.clear()
        return


class Synthesizer(sd.OutputStream):
    def __init__(self, amplitude: float, samplerate: int, blocksize: int,
                 deviceid: int or str, nchannels: int, dtype: np.dtype or str,
                 **kwargs):
        from platform import system
        if system() == 'Windows':
            kwargs['extra_settings'] = sd.WasapiSettings(True)
        super().__init__(samplerate, blocksize, deviceid, nchannels,
                         dtype, 'low', callback=self.callback, **kwargs)
        self._freq = mp.Value('f', float())
        self._omega = mp.Value('f', float())
        self.statuses = []
        self.envelope = Envelope()
        self.amplitude = amplitude
        self.timeDelta = (self.blocksize-1)/self.samplerate
        self.timeSpace = np.linspace(0, self.timeDelta, self.blocksize)
        self.note = 0.
        return

    def oscilator(self):
        out = self.amplitude * \
            (self.envelope.amplitude(self.timeSpace + self.time) *
                (oscilator('sine', self.freq, self.timeSpace + self.time) +
                 0.5*oscilator('warmsaw', self.freq, self.timeSpace + self.time)))
        return out

    def callback(self, outdata, frames, stime, status):
        for ch in range(self.channels):
            outdata[:, ch] = self.oscilator()
        if status:
            self.statuses.append(status)
        return

    @property
    def note(self):
        return self._note

    @note.setter
    def note(self, key: str):
        self._note = str(key).upper()
        self._freq.value, self._omega.value = get_note_freq(self.note)
        return

    @property
    def freq(self):
        return self._freq.value

    @property
    def omega(self):
        return self._omega.value
