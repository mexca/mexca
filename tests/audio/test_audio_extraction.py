""" Test voice feature extraction classes and methods """

import json
import os
import numpy as np
import pytest
import mexca.audio.features
from mexca.audio.extraction import VoiceExtractor
from mexca.core.exceptions import TimeStepError


class TestVoiceExtractor:
    extractor = VoiceExtractor(time_step=0.04)
    filepath = os.path.join('tests', 'test_files', 'test_eng_5_seconds.wav')

    with open(os.path.join('tests', 'reference_files', 'features_eng_5_seconds.json'),
              'r', encoding="utf-8") as file:
        reference_features = json.loads(file.read())


    def test_properties(self):
        with pytest.raises(ValueError):
            self.extractor.time_step = -1.0

        with pytest.raises(TypeError):
            self.extractor.time_step = 'k'

        with pytest.raises(TypeError):
            self.extractor.features = ['a']


    def test_extract_features(self):
        audio_features = self.extractor.extract_features(
            self.filepath,
            time=None
        )

        for feature in self.extractor.features:
            assert feature in audio_features and feature in self.reference_features
            assert pytest.approx(audio_features[feature], nan_ok=True) == np.array(
                self.reference_features[feature])


    def test_time_step_error(self):
        new_extractor = VoiceExtractor()
        with pytest.raises(TimeStepError):
            new_extractor.extract_features(
                self.filepath,
                time=None
            )


    def test_set_custom_features(self):
        features = {
            'pitchF0': mexca.audio.features.FeaturePitchF0()
        }
        new_extractor = VoiceExtractor(time_step=0.04, features=features)
        audio_features = new_extractor.extract_features(
            self.filepath,
            time=None
        )

        for feature in self.extractor.features:
            assert feature in audio_features and feature in self.reference_features
            assert pytest.approx(audio_features[feature], nan_ok=True) == np.array(
                self.reference_features[feature])
