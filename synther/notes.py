# -*- coding: utf-8 -*-


from numpy import pi, round


def calc_note_freq(refFreq, demiRatio, demiDiff):
    freq = round(refFreq*demiRatio**demiDiff, 6)
    omega = 2 * pi * freq
    return (freq, omega)


def get_note_freq(noteName: str = 'C0') -> tuple:
    noteName = noteName.upper()
    totalDemitones = 0

    # note
    note = noteName[0]
    if note == '0':
        return calc_note_freq(refNoteFreq, 0, 1)
    elif note in dictOfNotes:
        totalDemitones += dictOfNotes[note]

    # octave
    octave = int(noteName[-1])
    if octave in octaveRange:
        totalDemitones += demitonesInOctave*(octave-refOctave)

    # sharp, flat
    demitone = noteName.strip(note).strip(str(octave))
    if demitone in demitoneSymbols:
        totalDemitones += demitoneSymbols[demitone]

    return calc_note_freq(refNoteFreq, demitoneRatio, totalDemitones)


dictOfNotes = {'A': 9,
               'B': 11,
               'C': 0,
               'D': 2,
               'E': 4,
               'F': 5,
               'G': 7}


octaveRange = range(0, 9)
octaveRatio = 2

demitoneSymbols = {'#': 1, 'o': 0, 'b': -1}
demitonesInOctave = 12

demitoneRatio = octaveRatio**(1/demitonesInOctave)

refNote = 'C'
refOctave = 0
refNoteFreq, refNoteAngular = calc_note_freq(27.5, demitoneRatio, -9)  # C0


if __name__ == '__main__':
    print(get_note_freq('A4'))
    print(get_note_freq('G3'))
