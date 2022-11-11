# -*- coding: utf-8 -*-
from dataclasses import dataclass
from synther.keyparser import keyparser
from synther.instruments import Instrument
from numpy import ndarray


class Note:
    def __init__(self, key: str, instrument: Instrument, time_on: float) -> None:
        self.name = key
        self.frequency = keyparser.parse(key)
        self.instrument = instrument
        self.time_on = time_on
        self.time_off = -1.
        self.finished = False

    def make_sound(self, timestamps: ndarray) -> ndarray:
        envelope = self.instrument.envelope.amplitude(timestamps, self.time_on, self.time_off)
        sound = self.instrument.waveform(self.frequency, timestamps)
        if (envelope <= 0).all():
            self.finished = True
        return (envelope * sound).T
