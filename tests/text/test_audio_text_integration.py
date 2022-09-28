""" Test Audio text integration classes and methods """

import json
import os
import pytest
import numpy as np
from spacy.tokens import Doc
from mexca.core.exceptions import TimeStepError
from mexca.text.transcription import (
    AudioTextIntegrator, AudioTranscriber,
    TextRestaurator, SentimentExtractor
)

# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestTextRestaurator:

    # Use fixture to prevent class from being initialized when tests are skipped
    @pytest.fixture
    def restaurator(self):
        return TextRestaurator()

    def test_properties(self, restaurator):
        with pytest.raises(TypeError):
            restaurator.model = -1

        with pytest.raises(TypeError):
            restaurator.punctuator = 'k'

        with pytest.raises(TypeError):
            restaurator.sentencizer = 'k'


    def test_apply(self, restaurator):
        text = 'today is a good day yesterday was not a good day'

        transcription = {
            'transcription': text,
            'start_timestamps': np.arange(len(text)),
            'end_timestamps': np.arange(len(text))
        }

        text_restored = restaurator.apply(transcription)

        assert text_restored.text == 'today is a good day. yesterday was not a good day.'
        assert isinstance(text_restored, Doc)
        assert hasattr(text_restored, 'sents')


@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestAudioTextIntegrator:
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    # reference output
    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'),
        'r', encoding="utf-8") as file:
        text_audio_transcription = json.loads(file.read())

    @pytest.fixture
    def audio_text_integrator(self):
        return AudioTextIntegrator(
            audio_transcriber=AudioTranscriber(language='dutch'),
            text_restaurator=TextRestaurator(),
            sentiment_extractor=SentimentExtractor()
        )


    def test_properties(self, audio_text_integrator):
        with pytest.raises(TypeError):
            audio_text_integrator.audio_transcriber = 'k'

        with pytest.raises(TypeError):
            audio_text_integrator.text_restaurator = 'k'

        with pytest.raises(TypeError):
            audio_text_integrator.sentiment_extractor = 'k'

        with pytest.raises(ValueError):
            audio_text_integrator.time_step = -2.0

        with pytest.raises(TypeError):
            audio_text_integrator.time_step = 'k'


    def test_apply(self, audio_text_integrator):
        with pytest.raises(TypeError):
            out  = audio_text_integrator.apply(self.filepath, 'k')

        out  = audio_text_integrator.apply(self.filepath, self.text_audio_transcription['time'])
        assert pytest.approx(out['text_token_id'], nan_ok=True) == self.text_audio_transcription['text_token_id']
        assert [token in ['', 'maak', 'en', 'er', 'groen', 'als'] for token in out['text_token']]
        assert pytest.approx(out['text_token_start'], nan_ok=True) == self.text_audio_transcription['text_token_start']
        assert pytest.approx(out['text_token_end'], nan_ok=True) == self.text_audio_transcription['text_token_end']
        assert pytest.approx(out['text_sent_id'], nan_ok=True) == self.text_audio_transcription['text_sent_id']
        assert out['text_sent_pos'].shape == np.array(self.text_audio_transcription['text_sent_pos']).shape
        assert out['text_sent_neg'].shape == np.array(self.text_audio_transcription['text_sent_neg']).shape
        assert out['text_sent_neu'].shape == np.array(self.text_audio_transcription['text_sent_neu']).shape


    def test_apply_error(self, audio_text_integrator):
        with pytest.raises(TimeStepError):
            audio_text_integrator.apply(self.filepath, time=None)
