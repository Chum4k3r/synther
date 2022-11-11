# -*- coding: utf-8 -*-

class KeyParser:
    # Notes
    pitch_classes = {'A': 9,
                     'B': 11,
                     'C': 0,
                     'D': 2,
                     'E': 4,
                     'F': 5,
                     'G': 7}

    # Sharp or flat
    sharpness = {'#': 1,
                 '': 0,
                 'b': -1}

    ref_freq: float = 16.3516  # C0
    demitone_ratio: float = 2**(1/12)

    def parse(self, key: str) -> float:
        demitones = 0

        # note
        note = key[0]
        demitones += self.pitch_classes[note]

        # octave
        octave = key[-1]
        demitones += 12*int(octave)

        # sharp or flat
        sof = key.strip(note).strip(octave)
        demitones += self.sharpness[sof]

        return round(self.ref_freq * self.demitone_ratio**demitones, 6)


keyparser = KeyParser()
