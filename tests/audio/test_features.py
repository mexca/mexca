""" Test voice feature classes and methods """

import json
import os
import numpy as np
import pytest
from parselmouth import Sound
from mexca.audio.features import FeaturePitchF0


class TestFeaturePitchF0:
    feature = FeaturePitchF0()
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')
    snd = Sound(filepath)
    time_step = 0.04
    end_time = snd.get_end_time()
    time = np.arange(start=0.0, stop=end_time, step=time_step)

    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'),
        'r', encoding="utf-8"
    ) as file:
        reference = json.loads(file.read())['pitchF0']


    def test_properties(self):
        with pytest.raises(ValueError):
            self.feature.time_step = -1.0

        with pytest.raises(TypeError):
            self.feature.time_step = 'k'

        with pytest.raises(ValueError):
            self.feature.pitch_floor = -1.0

        with pytest.raises(TypeError):
            self.feature.pitch_floor = 'k'

        with pytest.raises(ValueError):
            self.feature.pitch_ceiling = -1.0

        with pytest.raises(TypeError):
            self.feature.pitch_ceiling = 'k'


    def test_extract(self):
        pitch = self.feature.extract(self.snd, self.time)
        assert pytest.approx(pitch, nan_ok=True) == np.array(self.reference)
