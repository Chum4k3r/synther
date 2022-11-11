# -*- coding: utf-8 -*-
from dataclasses import dataclass
from numpy import ndarray, zeros


class Envelope:
    def amplitude(self, timestamps: ndarray, time_note_on: float, time_note_off: float) -> ndarray:
        raise NotImplementedError("Must be implemented in subclasses")


@dataclass
class EnvelopeADSR(Envelope):
    attack_dur: float
    decay_dur: float
    release_dur: float
    attack_amp: float
    sustain_amp: float

    def amplitude(self, timestamps: ndarray, time_note_on: float, time_note_off: float) -> ndarray:
        amp = self._adsr_amplitude(atk_time=self.attack_dur,
                                   dec_time=self.decay_dur,
                                   rel_time=self.release_dur,
                                   atk_amp=self.attack_amp,
                                   sus_amp=self.sustain_amp,
                                   timestamps=timestamps,
                                   time_on=time_note_on,
                                   time_off=time_note_off)
        return amp

    def _adsr_amplitude(self,
                        atk_time: float, dec_time: float,
                        rel_time: float, atk_amp: float,
                        sus_amp: float, timestamps: ndarray,
                        time_on: float, time_off: float) -> ndarray:
        # Allocate memory
        amplitude = zeros(timestamps.shape)    # array of the final amplitude
        envelope_times = timestamps - time_on

        # Atack
        atk = envelope_times <= atk_time # index time interval on the attack phase
        if atk.any():
            # register attack amplitude
            amplitude[atk] = (envelope_times[atk] / atk_time) * atk_amp

        # Decay
        dec = envelope_times > atk_time  # index decay phase
        dec[dec] = envelope_times[dec] <= (atk_time + dec_time)
        if dec.any():
            # register decay amplitude
            amplitude[dec] = ((envelope_times[dec] - atk_time) / dec_time) * (sus_amp - atk_amp) + atk_amp

        # Sustain
        sus = envelope_times > atk_time + dec_time  # index sustain phase
        if sus.any():
            # register sustain amplitude
            amplitude[sus] = sus_amp  # register amplitude

        if time_on < time_off:  # Note off
            # Release amplitude
            amplitude[:] = (envelope_times / rel_time) * (0 - amplitude) + amplitude

        amplitude[amplitude <= 0.0001] = 0.0   # set too low values of amplitude to zero!
        return amplitude
