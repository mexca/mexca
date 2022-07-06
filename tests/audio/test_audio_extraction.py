""" Test voice feature extraction classes and methods """

import json
import os
import numpy as np
import pytest
from mexca.audio.extraction import VoiceExtractor


class TestVoiceExtractor:
    extractor = VoiceExtractor(time_step=0.04)
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    with open(os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'),
              'r', encoding="utf-8") as file:
        reference_features = json.loads(file.read())

    def test_extract_features(self):
        audio_features = self.extractor.extract_features(
            self.filepath, time=None)

        for feature in self.extractor.features:
            assert feature in audio_features and feature in self.reference_features
            assert pytest.approx(audio_features[feature], nan_ok=True) == np.array(
                self.reference_features[feature])
