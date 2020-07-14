# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:18:07 2020

@author: joaovitor
"""

import platform
import numpy as np
import numba as nb
import multiprocessing as mp
import sounddevice as sd
from .notes import get_note_freq


@nb.njit
def oscilator(oscType: str, omega: float, timeSpace: np.ndarray,
              time: float) -> np.ndarray:
    if oscType == 'sine':
        return np.sin(omega*(timeSpace + time))
    elif oscType == 'square':
        return 2.*(np.sin(omega*(timeSpace + time)) >= 0)-1.


_ = oscilator('square', np.pi, np.linspace(0, 63/44100, 64), 0)
del _


class Synthesizer(sd.OutputStream):
    def __init__(self, osctype: str, samplerate: int, blocksize: int, deviceid: int or str,
                 nchannels: int, dtype: np.dtype or str, **kwargs):
        from platform import system
        if system() == 'Windows':
            kwargs['extra_settings'] = sd.WasapiSettings(True)
        super().__init__(samplerate, blocksize, deviceid, nchannels,
                         dtype, 'low', callback=self.callback, **kwargs)
        self._freq = mp.Value('f', float())
        self._omega = mp.Value('f', float())
        self.statuses = []
        self.amplitude = 0.5
        self.oscType = osctype
        self.timeDelta = (self.blocksize-1)/self.samplerate
        self.timeSpace = np.linspace(0, self.timeDelta, self.blocksize)
        self.note = 0.
        return

    def oscilator(self):
        return oscilator(self.oscType, self.omega, self.timeSpace, self.time)

    def callback(self, outdata, frames, stime, status):
        for ch in range(self.channels):
            outdata[:, ch] = self.amplitude*self.oscilator()
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
