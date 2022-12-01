""" Test Audio text integration classes and methods """

import json
import os
import pytest
import numpy as np
from pyannote.core import Annotation
from mexca.core.exceptions import TimeStepError
from mexca.text.transcription import AudioTextIntegrator, AudioTranscriber, SentimentExtractor

# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs

@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestAudioTextIntegrator:
    filepath = os.path.join('tests', 'test_files', 'test_eng_5_seconds.wav')

    # Reference annotation
    with open(os.path.join(
            'tests', 'reference_files', 'annotation_eng_5_seconds.json'
        ), 'r', encoding="utf-8") as file:
        annotation = Annotation.from_json(json.loads(file.read()))

    # Reference output
    with open(
        os.path.join('tests', 'reference_files', 'reference_eng_5_seconds.json'),
        'r', encoding="utf-8") as file:
        reference = json.loads(file.read())


    @pytest.fixture
    def audio_text_integrator(self):
        return AudioTextIntegrator(
            audio_transcriber=AudioTranscriber(),
            sentiment_extractor=SentimentExtractor()
        )


    def test_properties(self, audio_text_integrator):
        with pytest.raises(ValueError):
            audio_text_integrator.time_step = -2.0


    def test_apply(self, audio_text_integrator):
        out  = audio_text_integrator.apply(self.filepath, self.annotation, self.reference['time'])

        for key in out:
            assert key in self.reference
            if key in self.reference:
                assert np.array(out[key]).shape == np.array(self.reference[key]).shape


    def test_apply_error(self, audio_text_integrator):
        with pytest.raises(TimeStepError):
            audio_text_integrator.apply(self.filepath, self.annotation, time=None)
