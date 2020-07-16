#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:18:07 2020

@author: joaovitor
"""

import numpy as _np
import numba as _nb
import multiprocessing as _mp
from pynput import keyboard as _kb
from sounddevice import OutputStream as _OutStream, \
    WasapiSettings as _Wasapi, default as _default


__all__ = ['Synthesizer', 'Envelope', 'Note']


@_nb.njit
def omega(freq: float) -> float:
    return 2*_np.pi*freq

_ = omega(1.)

@_nb.njit
def calc_note_freq(demiDiff: int, refFreq: float = 16.3516, demiRatio: float = 2**(1/12)) -> float:
    return _np.round(refFreq*demiRatio**demiDiff, 6)

_ = calc_note_freq(1)

@_nb.njit(parallel=True)
def oscilator(oscType: str, freq: float, timeSpace: _np.ndarray,
              LFOfreq: float = 0., LFOamp: float = 0.) -> _np.ndarray:
    theFreq = omega(freq)*timeSpace + \
        LFOamp*freq*_np.sin(omega(LFOfreq)*timeSpace)
    if oscType == 'sine':
        return _np.sin(theFreq)
    elif oscType == 'square':
        return 2. * (_np.sin(theFreq) >= 0) - 1.
    elif oscType == 'triangle':
        return _np.arcsin(_np.sin(theFreq)) * 2. / _np.pi
    elif oscType == 'warmsaw':
        out = _np.zeros((timeSpace.shape[0], ))
        for n in _nb.prange(1, 30):
            out[:] += (_np.sin(n * theFreq) / n)
        return out * (2 / _np.pi)
    elif oscType == 'sawtooth':
        return (2. / _np.pi) * ( freq * _np.pi * (timeSpace % (1/freq)) - (_np.pi/2.))
    elif oscType == 'noise':
        return _np.random.randn(timeSpace.shape[0])
    else:
        return _np.zeros((timeSpace.shape[0], ))

_ts = _np.linspace(0, 64/44100, 64)
_ = oscilator('warmsaw', _np.pi, _ts)


@_nb.njit
def envelope_amplitude(atkTime: float, decTime: float, relTime: float, atkAmp: float,
                       susAmp:float, timeSpace: _np.ndarray, timeOn: float, timeOff: float):
    amplitude = _np.zeros(timeSpace.shape[0])
    releaseAmp = _np.zeros(timeSpace.shape[0])
    atk = _np.zeros(timeSpace.shape[0])
    dec = _np.zeros(timeSpace.shape[0])
    sus = _np.zeros(timeSpace.shape[0])
    lifeTime = _np.zeros(timeSpace.shape[0])

    if timeOn > timeOff:  # Note on

        lifeTime = timeSpace - timeOn

        # Atack
        atk = lifeTime <= atkTime
        if atk.any():
            amplitude[atk] = (lifeTime[atk] / atkTime) * atkAmp

        # Decay
        dec = lifeTime > atkTime
        dec[dec] = lifeTime[dec] <= (atkTime + decTime)
        if dec.any():
            amplitude[dec] = ((lifeTime[dec] - atkTime) / decTime) * \
                (susAmp - atkAmp) + atkAmp

        # Sustain
        sus = lifeTime > atkTime + decTime
        if sus.any():
            amplitude[sus] = susAmp

    else:  # note off
        lifeTime = (timeOff - timeOn) - timeSpace

        # Release while on attack
        atk = lifeTime <= atkTime
        if atk.any():
            releaseAmp[atk] = (lifeTime[atk] / atkTime) * atkAmp

        # Release while on decay
        dec = lifeTime > atkTime
        dec[dec] = lifeTime[dec] <= (atkTime + decTime)
        if dec.any():
            releaseAmp[dec] = ((lifeTime[dec] - atkTime) / decTime) * \
                (susAmp - atkAmp) + atkAmp

        # Release while on sustain
        sus = lifeTime > atkTime + decTime
        if sus.any():
            releaseAmp[sus] = susAmp

        amplitude[:] = ((timeSpace - timeOff) / relTime) * \
            (0 - releaseAmp) + releaseAmp

    safe = amplitude <= 0.0001
    if safe.any():
        amplitude[safe] = 0.0

    return amplitude


_ = envelope_amplitude(0.1, 0.1, 0.1, 1., 0.8, _ts, 0.02, 0.)

@_nb.njit
def harmonica_sound(timeSpace: _np.ndarray, freq: float, timeOn: float):
    som = (oscilator('square', freq, timeSpace, 5., 0.005)
        + 0.5 * oscilator('square', 1.5*freq, timeSpace - timeOn)
        + 0.25 * oscilator('square', 3*freq, timeSpace - timeOn)
        + 0.05 * oscilator('noise', 0., timeSpace - timeOn))
    return som

_ = harmonica_sound(_ts, 1., 0.1)

del _





class Note:

    # Notes
    _notesDict = {'A': 9,
                  'B': 11,
                  'C': 0,
                  'D': 2,
                  'E': 4,
                  'F': 5,
                  'G': 7}

    # Sharp or flat
    _sofDict = {'#': 1,
                '': 0,
                'b': -1}

    _octaveRange = range(9)

    def __init__(self, name: str):
        self.name = str(name).upper()
        self.freq: float = self._get_note_freq(self.name)  # position in scale
        self.timeOn: float = 0.0  # time started
        self.timeOff: float = 0.0  # time stopped
        self.active: bool = False
        self.channel: int = 0  # something like midi channel
        return

    def __repr__(self):
        return f"Note('{self.name}')"

    def _get_note_freq(self, noteName: str = 'C0') -> float:
        totalDemitones = 0

        # note
        note = noteName[0]
        totalDemitones += self._notesDict[note]

        # octave
        octave = int(noteName[-1])
        totalDemitones += 12*octave

        # sharp or flat
        sof = noteName.strip(note).strip(str(octave))
        totalDemitones += self._sofDict[sof]

        return calc_note_freq(totalDemitones)


class Envelope:
    def __init__(self, attack: float, decay: float, release: float, start: float, sustain: float):
        self.attackTime = attack
        self.decayTime = decay
        self.releaseTime = release

        self.startAmp = start
        self.sustainAmp = sustain
        return

    def amplitude(self, timeSpace, timeOn, timeOff):
        return envelope_amplitude(self.attackTime, self.decayTime,
                                  self.releaseTime, self.startAmp,
                                  self.sustainAmp, timeSpace, timeOn, timeOff)

class Instrument:

    _instance = None

    def __init__(self, level: float, envelope: Envelope):
        self.level = level
        self.envelope = envelope
        return

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def sound(self, timeSpace: _np.ndarray, note: Note, noteFinished: _mp.Event) -> _np.ndarray:
        pass


class Bell(Instrument):
    def __init__(self, level: float = 0.8):
        envelope = Envelope(0.01, 1., 1., 1., 0.)
        super().__init__(level, envelope)
        return

    def sound(self, timeSpace: _np.ndarray, note: Note, noteFinished: _mp.Event) -> _np.ndarray:
        amplitude = self.envelope.amplitude(timeSpace, note.timeOn, note.timeOff)
        if amplitude.all() <= 0.0:
            noteFinished.set()

        # Glokenspiel
        sound = (oscilator('sine', 2*note.freq, timeSpace - note.timeOn, 5., 0.005)
            + 0.5 * oscilator('sine', 3*note.freq, timeSpace - note.timeOn)
            + 0.25 * oscilator('sine', 4*note.freq, timeSpace - note.timeOn))
        return self.level * amplitude * sound


class Harmonica(Instrument):
    def __init__(self, level: float = 0.6):
        envelope = Envelope(0.1, 0.05, 0.1, 1., 0.8)
        super().__init__(level, envelope)
        return

    def sound(self, timeSpace: _np.ndarray, note: Note, noteFinished: _mp.Event) -> _np.ndarray:
        amplitude = self.envelope.amplitude(timeSpace, note.timeOn, note.timeOff)
        if amplitude.all() <= 0.0:
            noteFinished.set()
        sound = harmonica_sound(timeSpace, note.freq, note.timeOn)
        return self.level * amplitude * sound


class Synthesizer:
    def __init__(self, amplitude: float, samplerate: int, blocksize: int):
        self.instruments = {1: Bell(), 2: Harmonica()}
        self.notes = {}
        self.noteFinished = _mp.Event()
        self.notesFree = _mp.Event()
        self.notesFree.set()

        self.samplerate = samplerate
        self.blocksize = blocksize

        timeDelta = (self.blocksize-1)/self.samplerate
        self.timeSpace = _np.linspace(0, timeDelta, self.blocksize)
        return

    def synthesize(self, time, nchannels):
        dTime = self.timeSpace + time
        output = _np.zeros((dTime.shape[0], nchannels))
        removes = []
        self.notesFree.clear()
        for name, note in self.notes.items():
            self.noteFinished.clear()
            for ch in range(nchannels):
                output[:, ch] = note.channel.sound(dTime, note, self.noteFinished)
            if self.noteFinished:
                note.active = False
                removes.append(name)
        self.notesFree.set()
        for name in removes:
            self.notes.pop(name)
        return output

    def note_names(self):
        return list(self.notes.keys())


class _App(Synthesizer):

    finished = _mp.Event()

    # TODO: Improve key map, add transposing
    key_map = {'z': 'C3', 's': 'C#3', 'x': 'D3', 'd': 'D#3',
               'c': 'E3', 'v': 'F3', 'g': 'F#3', 'b': 'G3',
               'h': 'G#3', 'n': 'A3', 'j': 'A#3', 'm': 'B3',
               ',': 'C4', 'e': 'C4', '4': 'C#4', 'r': 'D4',
               '5': 'D#4', 't': 'E4', 'y': 'F4', '7': 'F#4',
               'u': 'G4', '8': 'G#4', 'i': 'A4', '9': 'A#4',
               'o': 'B4', 'p': 'C5'}

    welcome = """
Welcome to the Synther!
Play using any of the following keys

   C3                          C4                          C5
  |  | | | |  |  | | | | | |  |  | | | |  |  | | | | | |  |  |
  |  |S| |D|  |  |G| |H| |J|  |  |4| |5|  |  |7| |8| |9|  |  |
  |  |_| |_|  |  |_| |_| |_|  | ,|_| |_|  |  |_| |_| |_|  |  |_
  | Z | X | C | V | B | N | M | E | R | T | Y | U | I | O | P |
  |___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|

Use ESC to quit.
"""

    _instance = None
    no_key = ' '

    def __init__(self, amplitude: float = 0.5, samplerate: int = 44100,
                 blocksize: int = 64):
        super().__init__(amplitude, samplerate, blocksize)
        return

    def __call__(self, deviceid: list = _default.device,
                 nchannels: int = 1, dtype: _np.dtype or str = 'float32',
                 **kwargs):
        from platform import system
        if system() == 'Windows':
            kwargs['extra_settings'] = _Wasapi(True)

        self.statuses = []

        with _OutStream(self.samplerate, self.blocksize, deviceid,
                          nchannels, dtype, latency='low', callback=self.callback,
                          finished_callback=self.callback_end,
                          **kwargs) as self.stream:
            with _kb.Listener(on_press=self.key_press,
                             on_release=self.key_release,
                             supress=True) as keys:
                print(self.welcome)
                self.print_func()
                keys.join()
        self.finished.wait()
        print()
        print("Goodbye!")
        return 0

    def print_func(self, note: Note = None):
        (print(f'\rLatency: {1e3*self.stream.latency:.3f} ms. Playing {note.name} at {note.freq} Hz', end='\r')
             if note is not None else
                 print(f'\rLatency: {1e3*self.stream.latency:.3f} ms. {self.no_key: <{28}}', end='\r'))
        return

    def callback(self, outdata, frames, stime, status):
        outdata[:] = self.synthesize(self.stream.time, self.stream.channels)
        if status:
            self.statuses.append(status)
        return


    def callback_end(self):
        if len(self.statuses) >= 1:
            print('\n')
            print('The following status were thrown by PortAudio during app execution:')
            [print(status) for status in self.statuses]
        self.finished.set()
        return

    def key_press(self, key):
        # print('press')
        if type(key) is _kb.Key:
            if key != _kb.Key.esc:
                return
            return False
        else:
            if key.char not in self.key_map:
                return
            noteName = self.key_map[key.char]
            if noteName not in self.notes:
                note = Note(noteName)
                note.timeOn = self.stream.time
                note.channel = self.instruments[2]
                note.active = True
                self.notesFree.wait()
                self.notes[noteName] = note
                # self.print_func(note)
            else:
                note = self.notes[noteName]
                if note.timeOff > note.timeOn:
                    note.timeOn = self.stream.time
                    note.active = True
        return

    def key_release(self, key):
        if type(key) is _kb.Key:
            return
        elif key.char not in self.key_map:
            return
        noteName = self.key_map[key.char]
        if noteName in self.notes:
            self.notes[noteName].timeOff = self.stream.time
        # self.print_func()
        return

bell = Bell()
harmonica = Harmonica()

if __name__ == '__main__':
    import sys

    app = _App()
    sys.exit(app((0, 0)))
