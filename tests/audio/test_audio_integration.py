""" Test audio speaker id and voice feature integration classes and methods """

import json
import os
import pytest
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.integration import AudioIntegrator
from mexca.audio.identification import SpeakerIdentifier


class TestAudioIntegrator:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    integrator = AudioIntegrator(
        SpeakerIdentifier(use_auth_token=use_auth_token),
        VoiceExtractor(time_step=0.04)
    )
    filepath = os.path.join('tests', 'test_files', 'test_eng_5_seconds.wav')

    with open(
        os.path.join('tests', 'reference_files', 'features_eng_5_seconds.json'), 'r', encoding="utf-8"
    ) as file:
        reference_features = json.loads(file.read())


    def test_properties(self):
        with pytest.raises(TypeError):
            self.integrator.identifier = 3.0

        with pytest.raises(TypeError):
            self.integrator.extractor = 3.0


    def test_integrate(self):
        with pytest.raises(TypeError):
            annotated_features = self.integrator.apply(self.filepath, 'k')

        with pytest.raises(TypeError):
            annotated_features = self.integrator.apply(self.filepath, None, show_progress='k')

        annotated_features = self.integrator.apply(self.filepath, None)
        assert all(annotated_features['segment_id'] == self.reference_features['segment_id'])
