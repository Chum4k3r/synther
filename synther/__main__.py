# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:45:11 2020

@author: joaovitor
"""

import sys
from pynput import keyboard as kb
from multiprocessing import Event
from synther.synthesizer import Synthesizer


class _MainLoop:

    finished = Event()

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

    def __init__(self):
        return

    def __call__(self, **kwargs):
        with Synthesizer(finished_callback=self.finished_callback,
                         **kwargs) as self.synth:
            with kb.Listener(on_press=self.key_press,
                             on_release=self.key_release,
                             supress=True) as keys:
                print(self.welcome)
                self.print_func()
                keys.join()
        self.finished.wait()
        print()
        print("Goodbye!")
        return 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def print_func(self):
        print(f'\rLatency: {1e3*self.synth.latency:.3f} ms. Playing {self.synth.note} at {self.synth.freq:.3f} Hz', end='\r') \
            if self.synth.freq > 0. \
            else print(f'\rLatency: {1e3*self.synth.latency:.3f} ms. {self.no_key: <{28}}', end='\r')
        return

    def finished_callback(self):
        if len(self.synth.statuses) >= 1:
            print('\n')
            print('The following status were thrown by PortAudio during app execution:')
            [print(status) for status in self.synth.statuses]
        self.finished.set()
        return

    def key_press(self, key):
        if type(key) is kb.Key:
            if key != kb.Key.esc:
                return
            return False
        else:
            if key.char not in self.key_map:
                return
            self.synth.note = self.key_map[key.char]
            self.synth.envelope.note_on(self.synth.time)
        self.print_func()
        return

    def key_release(self, key):
        self.synth.envelope.note_off(self.synth.time)
        self.synth.note = 0.
        self.print_func()
        return


ml = _MainLoop()
sys.exit(ml(amplitude=0.6, samplerate=44100, blocksize=144, deviceid=(0, 0), nchannels=2, dtype='float32', clip_off=False))
