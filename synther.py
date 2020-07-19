#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesizer.

Created on Sat Jul 11 23:18:07 2020

@author: joaovitor
"""

import numpy as _np
import numba as _nb
import multiprocessing as _mp
import sounddevice as _sd
# import soundcard as _sc
import time as _time
from pynput import keyboard as _kb


__all__ = ['Synthesizer', 'Envelope', 'Note', 'Instrument', 'oscilator']


loadTime = _time.time()


# %% API

class Envelope:
    """An instrument envelope interface."""

    def __init__(self, attack: float, decay: float, release: float, start: float, sustain: float):
        """
        Envelope is responsible for generating the instrument amplitude at a given time instant.

        A call to `.amplitude()` with a time interval and a `Note`s `.timeOn` and `.timeOff` parameters,
        returns the instrument waveform's amplitude for the given note.

        Parameters
        ----------
        attack : float
            Attack time duration.
        decay : float
            Decay time duration.
        release : float
            Release time duration.
        start : float
            Maximum amplitude at the end of attack phase.
        sustain : float
            Sustain amplitude after decay phase.

        Returns
        -------
        None.

        """
        self.attackTime = attack
        self.decayTime = decay
        self.releaseTime = release

        self.startAmp = start
        self.sustainAmp = sustain
        return

    def amplitude(self, timeSpace: _np.ndarray, noteTimeOn: float, noteTimeOff: float) -> _np.ndarray:
        """
        Return the amplitude in a given time interval.

        The time interval is based on how much time has passed since the note has been played, or released.

        Parameters
        ----------
        timeSpace : _np.ndarray
            Envelope time interval.
        noteTimeOn : float
            Time instant of note pressed.
        noteTimeOff : float
            Time instant of note released.

        Returns
        -------
        amp : _np.ndarray
            DESCRIPTION.

        """
        amp = _envelope_amplitude(self.attackTime, self.decayTime,
                                  self.releaseTime, self.startAmp,
                                  self.sustainAmp, timeSpace,
                                  noteTimeOn, noteTimeOff)
        return amp


class Instrument:
    """Instrument interface class. Must be subclassed."""

    _instance = None

    def __init__(self, level: float, envelope: Envelope):
        self.level = level
        self.envelope = envelope
        return

    def __new__(cls):
        """Singleton class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def sound(self, timeSpace, note):
        """
        Generate the actual instrument sound wave, based on the time interval and a note to play.

        This method must be overriden.

        Parameters
        ----------
        timeSpace : _np.ndarray
            The time interval.
        note : Note
            A note playing.

        Returns
        -------
        None.

        """
        pass


class Note:
    """Notes interface."""

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

    def __init__(self, name: str, instrument: Instrument, timeOn: float):
        """
        Represent a note being played, have a frequency and a name.

        Parameters
        ----------
        name : str
            Note name and octave, e.g. 'A4', 'C3'.
        timeOn : float
            Time instante where it was played.
        channel : int
            The instrument channel that played it.

        Returns
        -------
        None.

        """
        self.name = name
        self.channel = instrument  # something like midi channel
        self.timeOn = timeOn  # time started
        self.timeOff: float = 0.0  # time stopped
        self.freq: float = self._get_note_freq(self.name)
        self._finished = _mp.Event()
        return

    def __repr__(self):
        """Note representation."""
        return f"Note({str(self.name)}, {self.instrument}, {self.timeOn})"

    def _get_note_freq(self, noteName: str) -> float:
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

        return _calc_note_freq(totalDemitones)

    @property
    def finished(self):
        """Wether the note is finished or not."""
        return self._finished.is_set()

    @finished.setter
    def finished(self, tf):
        if not tf:
            self._finished.clear()
        else:
            self._finished.set()
        return


class Bell(Instrument):
    """A Bell."""

    def __init__(self, level: float = 0.8):
        """Bell like sounding instrument."""
        envelope = Envelope(0.01, 1., 1., 1., 0.)
        super().__init__(level, envelope)
        return

    def sound(self, timeSpace: _np.ndarray, note: Note) -> _np.ndarray:
        """
        Create the samples of a bell sound for a given `timeSpace` and `note`.

        Parameters
        ----------
        timeSpace : _np.ndarray
            DESCRIPTION.
        note : Note
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        amplitude = self.envelope.amplitude(timeSpace, note.timeOn, note.timeOff)
        if amplitude.all() <= 0.0:
            note.finished = True
        sound = _bell_sound(timeSpace, note.freq, note.timeOn)
        return self.level * amplitude * sound


class Harmonica(Instrument):
    """A Harmonica."""

    def __init__(self, level: float = 0.6):
        """Harmonica like sounding instrument."""
        envelope = Envelope(0.1, 0.05, 0.1, 1., 0.8)
        super().__init__(level, envelope)
        return

    def sound(self, timeSpace: _np.ndarray, note: Note) -> _np.ndarray:
        """
        Create the smaples of a harmonica like sounding instrument.

        Parameters
        ----------
        timeSpace : _np.ndarray
            DESCRIPTION.
        note : Note
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        amplitude = self.envelope.amplitude(timeSpace, note.timeOn, note.timeOff)
        if amplitude.all() <= 0.0:
            note.finished = True
        sound = _harmonica_sound(timeSpace, note.freq, note.timeOn)
        return self.level * amplitude * sound


class Synthesizer:
    """A synthesizer."""

    def __init__(self, samplerate: int, blocksize: int, nchannels: int):
        """
        Synthesizer.

        Parameters
        ----------
        samplerate : int
            DESCRIPTION.
        blocksize : int
            DESCRIPTION.
        nchannels : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.nchannels = nchannels

        self.instrument = instrument

        self.playingNotes = {}
        self.freeToAddNotes = _mp.Event()
        self.freeToAddNotes.set()

        _timeDelta = (self.blocksize-1)/self.samplerate
        self.timeSpace = _np.linspace(0., _timeDelta, self.blocksize, dtype='float32')

        self.initTime = _time.time()
        return

    @property
    def time(self):
        """Time since the synthesizer started.."""
        return _time.time() - self.initTime

    def get_samples(self):
        """
        Generate the audio samples for all notes being played during the current time interval.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        # Generate a time interval and allocate memory for the output
        timeSpace = self.timeSpace + self.time
        output = _np.zeros((self.blocksize, self.nchannels))

        finishedNotes = []  # Helper list to now which notes are finished

        self.freeToAddNotes.clear()  # prevent .add_note() modify the playingNotes dict
        for name, note in self.playingNotes.items():
            for ch in range(self.nchannels):
                # channels are manually filled because instruments
                # generates their sound as a 1d array
                # otherwise the nchannels information would have to
                # travel to the base _oscilator function.
                output[:, ch] += note.channel.sound(timeSpace, note)
            if note.finished:
                finishedNotes.append(name)
        self.freeToAddNotes.set()    # notes can be added again.

        # Remove finished notes
        [self.playingNotes.pop(name) for name in finishedNotes]
        return output

    def add_note(self, noteName: str, instrument: Instrument):
        """
        Add a new note to the `Synthesizer`'s `playingNotes`.

        Parameters
        ----------
        noteName : str
            DESCRIPTION.
        instrument : Instrument
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if noteName not in self.playingNotes:
            # Note is not beying played
            note = Note(noteName, instrument, self.time)
            self.freeToAddNotes.wait()  # wait untill it is safe to modify the playingNotes dict
            self.playingNotes[noteName] = note
        else:
            # Note is playing
            note = self.playingNotes[noteName]
            if note.timeOff > note.timeOn:
                # note has been played again during release phase.
                note.timeOn = self.time
        return

    def register_note_timeoff(self, noteName: str):
        """Register the time of a note being released."""
        if noteName in self.playingNotes:
            self.playingNotes[noteName].timeOff = self.time
        return


# %% NUMBA COMPILED STUFF

_ts = _np.linspace(0, 64/44100, 64)  # a time interval


@_nb.njit
def _omega(freq: float) -> float:
    """Return the angular frequency in rad/s for a given `freq` in Hertz."""
    return 2*_np.pi*freq


@_nb.njit(parallel=True)
def oscilator(oscType: str, freq: float, timeSpace: _np.ndarray,
              LFOfreq: float = 0., LFOamp: float = 0.) -> _np.ndarray:
    """
    Generate samples of an `oscType` wave form, for a given `timeSpace`.

    The amount of samples is equal to `timeSpace.shape`. The possible values of `oscType`s are:

        * 'sine'
        * 'square'
        * 'triangle'
        * 'warmsaw'
        * 'sawtooth'
        * 'noise'

    If `oscType` is not one of these values, silence (zeros) is returned.

    It is possible to generate frequency modulated waves by passing the `LFOfreq` and `LFOamp` arguments
    as non-zero values.

    Parameters
    ----------
    oscType : str
        A string with the name of the desired oscilator.
    freq : float
        The wave frequency, in Hertz.
    timeSpace : _np.ndarray
        The time interval to generate the samples.
    LFOfreq : float, optional
        The frequency of the modulation. The default is 0..
    LFOamp : float, optional
        Amplitude of the modulation. The default is 0..

    Returns
    -------
    _np.ndarray
        A numpy array populated with the wave samples for the given time interval.

    """
    theFreq = _omega(freq) * timeSpace + LFOamp * freq * _np.sin(_omega(LFOfreq) * timeSpace)
    out = _np.zeros(timeSpace.shape)

    if oscType == 'sine':
        out[:] += _np.sin(theFreq)

    elif oscType == 'square':
        out[:] += 2. * (_np.sin(theFreq) >= 0) - 1.

    elif oscType == 'triangle':
        out[:] += _np.arcsin(_np.sin(theFreq)) * 2. / _np.pi

    elif oscType == 'warmsaw':
        for n in _nb.prange(1, 30):
            out[:] += (_np.sin(n * theFreq) / n)
        return out * (2 / _np.pi)

    elif oscType == 'sawtooth':
        return (2. / _np.pi) * (freq * _np.pi * (timeSpace % (1/freq)) - (_np.pi/2.))

    elif oscType == 'noise':
        out[:] += _np.random.standard_normal(timeSpace.shape)
        out[:] /= _np.max(_np.abs(out))

    return out


_ = oscilator('sine', _np.pi, _ts)  # compile on instrument sound
_ = oscilator('square', _np.pi, _ts)  # compile on instrument sound
_ = oscilator('triangle', _np.pi, _ts)  # compile on instrument sound
_ = oscilator('warmsaw', _np.pi, _ts)  # compile on instrument sound
_ = oscilator('sawtooth', _np.pi, _ts)  # compile on instrument sound
_ = oscilator('noise', 0., _ts)  # compile on instrument sound


@_nb.njit
def _calc_note_freq(semiDiff: int, refFreq: float = 16.3516,
                    semiRatio: float = 2**(1/12)) -> float:
    """Return the frequency of a note based on semitone difference."""
    return _np.round(refFreq*semiRatio**semiDiff, 6)


_ = _calc_note_freq(1)  # compile


@_nb.njit
def _envelope_amplitude(atkTime: float, decTime: float, relTime: float,
                        atkAmp: float, susAmp: float, timeSpace: _np.ndarray,
                        timeOn: float, timeOff: float):
    """Envelope's amplitude calculation, see `Envelope.amplitude`."""
    # Allocate memory
    amplitude = _np.zeros(timeSpace.shape)    # array of the final amplitude
    releaseAmp = _np.zeros(timeSpace.shape)   # amplitude during release phase
    lifeTime = _np.zeros(timeSpace.shape)     # time interval of the envelope

    if timeOn > timeOff:  # Note on

        lifeTime[:] = timeSpace - timeOn  # time interval relative only to envelope

        # Atack
        atk = lifeTime <= atkTime  # index time interval on the attack phase
        if atk.any():
            amplitude[atk] = (lifeTime[atk] / atkTime) * atkAmp  # register amplitude

        # Decay
        dec = lifeTime > atkTime
        dec[dec] = lifeTime[dec] <= (atkTime + decTime)  # index decay phase
        if dec.any():
            # register amplitude
            amplitude[dec] = ((lifeTime[dec] - atkTime) / decTime) * \
                (susAmp - atkAmp) + atkAmp

        # Sustain
        sus = lifeTime > atkTime + decTime  # index sustain phase
        if sus.any():
            amplitude[sus] = susAmp  # register amplitude

    else:  # Note off
        lifeTime[:] = timeSpace - timeOn

        # Release while on attack
        atk = lifeTime <= atkTime
        if atk.any():
            releaseAmp[atk] = (lifeTime[atk] / atkTime) * atkAmp

        # Release while on decay
        dec = lifeTime > atkTime
        dec[dec] = lifeTime[dec] <= (atkTime + decTime)
        if dec.any():
            releaseAmp[dec] = ((lifeTime[dec] - atkTime) / decTime) * (susAmp - atkAmp) + atkAmp

        # Release while on sustain
        sus = lifeTime > atkTime + decTime
        if sus.any():
            releaseAmp[sus] = susAmp

        amplitude[:] = (lifeTime / relTime) * (0 - releaseAmp) + releaseAmp

    safe = amplitude <= 0.0001  # index amplitudes too small to be heard
    if safe.any():
        amplitude[safe] = 0.0   # set them to zero!

    return amplitude


_ = _envelope_amplitude(0.1, 0.1, 0.1, 1., 0.8, _ts, 0.02, 0.)  # compile


@_nb.njit(parallel=True)
def _harmonica_sound(timeSpace: _np.ndarray, freq: float, timeOn: float):
    """Compiled function to generate a Harmonica sound based on wave addition."""
    som = (1. * oscilator('warmsaw', 0.5*freq, timeSpace, 5., 0.005)
           + 0.5 * oscilator('square', freq, timeSpace - timeOn)
           + 0.25 * oscilator('square', 1.5*freq, timeSpace - timeOn)
           + 0.15 * oscilator('square', 3*freq, timeSpace - timeOn)
           + 0.05 * oscilator('noise', 0., timeSpace - timeOn))
    return som


_ = _harmonica_sound(_ts, 1., 0.1)  # compile


@_nb.njit(parallel=True)
def _bell_sound(timeSpace: _np.ndarray, freq: float, timeOn: float):
    """Compiled function to generate a Bell sound based on wave addition."""
    som = (1. * oscilator('sine', 2*freq, timeSpace)
           + 0.5 * oscilator('sine', 3*freq, timeSpace)
           + 0.25 * oscilator('sine', 4*freq, timeSpace))
    return som


_ = _bell_sound(_ts, 1., 0.1)  # compile

del _, _ts

loadTime = _time.time() - loadTime


# %% __main__

class Synther(Synthesizer):
    """Main Synther app."""

    statuses = []
    keysPressed = []
    stopStream = _mp.Event()

    finished = _mp.Event()

    # TODO: Improve key map, add transposing
    noteFromKey = {'z': 'C3', 's': 'C#3', 'x': 'D3', 'd': 'D#3',
                   'c': 'E3', 'v': 'F3', 'g': 'F#3', 'b': 'G3',
                   'h': 'G#3', 'n': 'A3', 'j': 'A#3', 'm': 'B3',
                   ',': 'C4', 'e': 'C4', '4': 'C#4', 'r': 'D4',
                   '5': 'D#4', 't': 'E4', 'y': 'F4', '7': 'F#4',
                   'u': 'G4', '8': 'G#4', 'i': 'A4', '9': 'A#4',
                   'o': 'B4', 'p': 'C5'}

    _instance = None
    no_key = ' '

    def __init__(self, loadTime: float, instrument: str, samplerate: int = 44100,
                 blocksize: int = 1536, nchannels: int = 2, FPS: int = 30):
        super().__init__(samplerate, blocksize, nchannels)
        if instrument == 'bell':
            inst = Bell()
        elif instrument == 'harmonica':
            inst = Harmonica()
        self.instrument = inst
        self.msPF = int(_np.ceil(1000/FPS))
        self.loadTime = loadTime
        return

    def __call__(self, deviceid: list = _sd.default.device,
                 dtype: _np.dtype or str = 'float32', **kwargs):
        from platform import system
        if system() == 'Windows':
            kwargs['extra_settings'] = _sd.WasapiSettings(True)

        with _sd.OutputStream(self.samplerate, self.blocksize, deviceid,
                              self.nchannels, dtype, latency='low', callback=self.callback,
                              finished_callback=self.callback_end, **kwargs) as stream:

            with _kb.Listener(on_press=self.key_press, on_release=self.key_release,
                              supress=True) as keys:

                print(self.welcome())
                while stream.active:
                    _sd.sleep(self.msPF)
                    if self.stopStream.is_set():
                        break
                    print(self.loop_string(stream), end='\r')
                keys.join()

        self.finished.wait()
        print()
        print("Goodbye!")
        return 0

    def welcome(self):
        s = f"""
Welcome to the Synther!
It took {self.loadTime:.3f} s to start.
Play using any of the following keys

   C3                          C4                          C5
  |  | | | |  |  | | | | | |  |  | | | |  |  | | | | | |  |  |
  |  |S| |D|  |  |G| |H| |J|  |  |4| |5|  |  |7| |8| |9|  |  |
  |  |_| |_|  |  |_| |_| |_|  | ,|_| |_|  |  |_| |_| |_|  |  |_
  | Z | X | C | V | B | N | M | E | R | T | Y | U | I | O | P |
  |___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|

Use ESC to quit.
"""
        return s

    def loop_string(self, stream):
        s = f'\rNotes playing: {len(self.playingNotes)}. CPU usage: {stream.cpu_load * 100:.2f}%{self.no_key: <12}'
        return s

    def callback(self, outdata, frames, stime, status):
        outdata[:] = self.get_samples()
        if status:
            self.statuses.append(status)
        if self.stopStream.is_set():
            raise _sd.CallbackStop
        return

    def callback_end(self):
        if len(self.statuses) >= 1:
            print('\n')
            print('The following status were thrown by PortAudio during app execution:')
            [print(status) for status in self.statuses]
        self.finished.set()
        return

    def key_press(self, key):
        if type(key) is _kb.Key:
            if key != _kb.Key.esc:
                return
            self.stopStream.set()
            return False
        else:
            if key.char not in self.noteFromKey:
                return
            if key.char not in self.keysPressed:
                self.add_note(self.noteFromKey[key.char], self.instrument)
                self.keysPressed.append(key.char)
        return

    def key_release(self, key):
        if type(key) is _kb.Key:
            return
        elif key.char not in self.noteFromKey:
            return
        self.register_note_timeoff(self.noteFromKey[key.char])
        self.keysPressed.remove(key.char)
        return


if __name__ == '__main__':
    import sys

    try:
        instrument = sys.argv[1]
    except IndexError:
        instrument = 'harmonica'

    synther = Synther(loadTime, instrument)

    sys.exit(synther(deviceid=10))
