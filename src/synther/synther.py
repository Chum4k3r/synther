# -*- coding: utf-8 -*-
import time as timing
from statistics import mean, median
from typing import Dict, Union
from multiprocessing import Event, Queue, Process
import numpy as np
from pynput import keyboard
import sounddevice as sd

from synther.instruments import Instrument
from synther.note import Note


class Synther:
    # TODO: Improve key map, add transposing
    note_key_map = {'z': 'C3', 's': 'C#3', 'x': 'D3', 'd': 'D#3',
                    'c': 'E3', 'v': 'F3', 'g': 'F#3', 'b': 'G3',
                    'h': 'G#3', 'n': 'A3', 'j': 'A#3', 'm': 'B3',
                    ',': 'C4', 'e': 'C4', '4': 'C#4', 'r': 'D4',
                    '5': 'D#4', 't': 'E4', 'y': 'F4', '7': 'F#4',
                    'u': 'G4', '8': 'G#4', 'i': 'A4', '9': 'A#4',
                    'o': 'B4', 'p': 'C5'}

    _instance = None
    no_key = ' '
    keys_pressed = []

    def __init__(self, samplerate: int, buffer_size: int, num_buffers: int, instrument: Instrument) -> None:
        self.instrument = instrument
        self.samplerate = samplerate
        self.delta_time = 1 / samplerate
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        self.buffer_duration = buffer_size / samplerate
        self.num_samples = num_buffers * buffer_size
        self.app_start = -1.
        self.active_notes: Dict[str, Note] = {}
        self.active = Event()
        self._safe_play_notes = Event()
        self._safe_play_notes.set()
        self.q = Queue()
        self.device = 0

    def elapsed_time(self):
        return (timing.perf_counter() - self.app_start)

    def play_note(self, key: str, instrument: Instrument) -> None:
        if key not in self.active_notes:
            note = Note(key, instrument, time_on=self.elapsed_time())
        else:
            note = self.active_notes[key]
            if note.time_off > note.time_on:
                note.time_on = self.elapsed_time()
        self._safe_play_notes.wait()
        self.active_notes[key] = note
        return

    def loop_string(self, notesPlaying, cpuLoad):
        s = f'\rNotes playing: {notesPlaying}. Sound generation time: {cpuLoad:.9f}{self.no_key: <12}\r'
        return s

    def run(self) -> None:
        self.app_start = timing.perf_counter()
        processing_times = []

        last_timestamp = 0.

        audio_loop_kwargs = dict(
            samplerate=self.samplerate,
            buffer_size=self.buffer_size,
            device=self.device,
            queue=self.q,
            active=self.active
        )
        process = Process(name="SyntherAudioProcess", target=self.audio_loop, kwargs=audio_loop_kwargs)

        process.start()
        with keyboard.Listener(on_press=self.key_press,
                               on_release=self.key_release,
                               suppress=True) as keys:

            print(self.welcome())

            self.active.set()
            while self.active.is_set():
                elapsed = self.elapsed_time()

                if elapsed < last_timestamp:
                    timing.sleep(last_timestamp - elapsed - self.buffer_duration)

                process_start = timing.perf_counter()
                sound = self.get_samples(elapsed)
                [self.q.put_nowait(sound[n:self.buffer_size:(n+1)*self.buffer_size]) for n in range(self.num_buffers)]
                process_duration = timing.perf_counter() - process_start

                processing_times.append(process_duration)
                print(self.loop_string(len(self.active_notes), process_duration), end='\r')

            keys.join()
        process.join()

        while True:
            try:
                data = self.q.get_nowait()
                print(data)
            except Exception as e:
                print(f"{type(e).__name__}")
                break

        print(f"{median(processing_times)=}")
        print(f"{mean(processing_times)=}")
        print(f"{min(processing_times)=}")
        print(f"{max(processing_times)=}")
        print(f"{self.buffer_duration=}")

        return

    def get_samples(self, elapsed: float):
        timestamps = elapsed + np.arange(self.num_samples) * self.delta_time
        sound = np.zeros((self.num_samples, 2))
        finished_notes = []

        self._safe_play_notes.clear()
        for name, note in self.active_notes.items():
            sound[:] += note.make_sound(timestamps)
            if note.finished:
                finished_notes.append(name)
        self._safe_play_notes.set()

        sound /= np.max(np.abs(sound), axis=0)

        [self.active_notes.pop(note) for note in finished_notes]
        return sound


    def welcome(self):
        s = f"""
Welcome to the Synther!
vPlay using any of the following keys
   C3                          C4                          C5
  |  | | | |  |  | | | | | |  |  | | | |  |  | | | | | |  |  |
  |  |S| |D|  |  |G| |H| |J|  |  |4| |5|  |  |7| |8| |9|  |  |
  |  |_| |_|  |  |_| |_| |_|  | ,|_| |_|  |  |_| |_| |_|  |  |_
  | Z | X | C | V | B | N | M | E | R | T | Y | U | I | O | P |
  |___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
Use ESC to quit.
"""
        return s

    def audio_loop(self, samplerate: int, buffer_size: int, device: int, queue: Queue, active: Event) -> None:

        finished = Event()
        statuses = []
        def callback(outdata, frames, stime, status):
            if status:
                statuses.append(status)
            try:
                outdata[:] = queue.get_nowait()
            except Exception as e:
                if type(e).__name__ == "Empty":
                    if not active.is_set():
                        raise sd.CallbackStop
            return

        active.wait()
        with sd.OutputStream(samplerate=samplerate,
                             blocksize=buffer_size,
                             device=device,
                             channels=2,
                             dtype='float32',
                             latency='low',
                             callback=callback,
                             finished_callback=finished.set) as stream:
            finished.wait()

        queue.put_nowait(statuses)
        return

    def key_press(self, key: Union[keyboard.Key, keyboard.KeyCode]):
        if isinstance(key, keyboard.Key):
            if key == keyboard.Key.esc:
                self.active.clear()
                return False
        elif isinstance(key, keyboard.KeyCode):
            if (key.char in self.note_key_map) and (key.char not in self.keys_pressed):
                self.play_note(self.note_key_map[key.char], self.instrument)
                self.keys_pressed.append(key.char)
        return

    def key_release(self, key: Union[keyboard.Key, keyboard.KeyCode]):
        if isinstance(key, keyboard.Key):
            return
        elif key.char not in self.note_key_map:
            return
        if (pitch := self.note_key_map[key.char]) in self.active_notes:
            self.active_notes[pitch].time_off = self.elapsed_time()
        self.keys_pressed.remove(key.char)
        return
