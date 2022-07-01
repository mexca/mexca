""" Test voice feature classes and methods """

import json
import os
import numpy as np
import pytest
from mexca.audio.features import FeaturePitchF0
from parselmouth import Sound

class TestFeaturePitchF0:
    feature = FeaturePitchF0()
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')
    snd = Sound(filepath)
    time_step = 0.04
    end_time = snd.get_end_time()
    time = np.linspace(start=0.0, stop=end_time, num=int(end_time/time_step))

    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'), 'r'
    ) as file:
        reference = json.loads(file.read())['pitchF0']

    def test_extract(self):
        pitch = self.feature.extract(self.snd, self.time)
        assert pytest.approx(pitch, nan_ok=True) == np.array(self.reference)
