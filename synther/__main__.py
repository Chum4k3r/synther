# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:45:11 2020

@author: joaovitor
"""

import sys
from time import sleep
from pynput import keyboard as kb
from multiprocessing import Event
from .synther import Synthesizer


class _MainLoop:

    FPS = 60
    dTime = 1/FPS

    finished = Event()
    running = Event()

    # TODO: Improve key map, add transposing
    key_map = {'z': 'C2', 's': 'C#2', 'x': 'D2', 'd': 'D#2',
        'c': 'E2', 'v': 'F2', 'g': 'F#2', 'b': 'G2', 'h': 'G#2',
        'n': 'A2', 'j': 'A#2', 'm': 'B2', 'e': 'C3', '4': 'C#3',
        'r': 'D3',  '5': 'D#3', 't': 'E3', 'y': 'F3', '7': 'F#3',
        'u': 'G3', '8': 'G#3', 'i': 'A3', '9': 'A#3', 'o': 'B3', 'p': 'C4'}

    welcome = """
Welcome to the Synther!
Play using any of the following keys

     C2                          C3                          C4
    |  | | | |  |  | | | | | |  |  | | | |  |  | | | | | |  |  |
    |  |S| |D|  |  |G| |H| |J|  |  |4| |5|  |  |7| |8| |9|  |  |
    |  |_| |_|  |  |_| |_| |_|  |  |_| |_|  |  |_| |_| |_|  |  |_
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
                self.running.set()
                nextTime = self.synth.time + self.dTime

                while self.running.is_set():
                    sleepTime = nextTime - self.synth.time
                    if sleepTime > 0.:
                        sleep(sleepTime)
                    self.print_func()
                    nextTime += self.dTime
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
        print(f'\rLatency: {self.synth.latency:.3f} s. Playing {self.synth.note} at {self.synth.freq:.3f} Hz', end='\r') \
            if self.synth.freq > 0. \
            else print(f'\rLatency: {self.synth.latency:.3f} s. {self.no_key: <{28}}', end='\r')
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
            self.running.clear()
            return False
        else:
            if key.char not in self.key_map:
                return
            self.synth.note = self.key_map[key.char]
            return

    def key_release(self, key):
        self.synth.note = 0.
        return


ml = _MainLoop()

# if __name__ == '__main__':
sys.exit(ml(osctype='square'))
