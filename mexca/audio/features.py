""" Voice feature classes and methods """

import numpy as np
from dataclasses import dataclass


class FeaturePitchF0:
    def __init__(self,
        time_step=None,
        pitch_floor=75.0,
        pitch_ceiling=600.0
    ) -> 'FeaturePitchF0':
        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling


    def extract(self, snd, time):
        voice_pitch = snd.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling
        )

        feature = np.vectorize(voice_pitch.get_value_at_time)(time)

        return feature
