""" Test audio speaker id and voice feature integration classes and methods """

import json
import numpy as np
import os
import pytest
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.integration import AudioIntegrator
from mexca.audio.speaker_id import SpeakerIdentifier

class TestAudioIntegrator:
    integrator = AudioIntegrator(
        SpeakerIdentifier(),
        VoiceExtractor(time_step=0.04)
    )
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'), 'r'
    ) as file:
        reference_features = json.loads(file.read())

    def test_integrate(self):
        annotated_features = self.integrator.apply(self.filepath, self.reference_features['time'])
        assert all(int(annotated_features['segment_id']) == int(self.reference_features['segment_id']))
