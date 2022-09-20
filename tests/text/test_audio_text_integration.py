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

class TestTextRestaurator:
    restaurator = TextRestaurator()

    def test_properties(self):
        with pytest.raises(TypeError):
            self.restaurator.model = -1

        with pytest.raises(TypeError):
            self.restaurator.punctuator = 'k'

        with pytest.raises(TypeError):
            self.restaurator.sentencizer = 'k'


    def test_apply(self):
        text = 'today is a good day yesterday was not a good day'

        transcription = {
            'transcription': text,
            'start_timestamps': np.arange(len(text)),
            'end_timestamps': np.arange(len(text))
        }

        text_restored = self.restaurator.apply(transcription)

        assert text_restored.text == 'today is a good day. yesterday was not a good day.'
        assert isinstance(text_restored, Doc)
        assert hasattr(text_restored, 'sents')


class TestAudioTextIntegrator:
    audio_text_integrator = AudioTextIntegrator(
        audio_transcriber=AudioTranscriber(language='dutch'),
        text_restaurator=TextRestaurator(),
        sentiment_extractor=SentimentExtractor()
    )
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    # reference output
    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'),
        'r', encoding="utf-8") as file:
        text_audio_transcription = json.loads(file.read())


    def test_properties(self):
        with pytest.raises(TypeError):
            self.audio_text_integrator.audio_transcriber = 'k'

        with pytest.raises(TypeError):
            self.audio_text_integrator.text_restaurator = 'k'

        with pytest.raises(TypeError):
            self.audio_text_integrator.sentiment_extractor = 'k'

        with pytest.raises(ValueError):
            self.audio_text_integrator.time_step = -2.0

        with pytest.raises(TypeError):
            self.audio_text_integrator.time_step = 'k'


    def test_apply(self):
        with pytest.raises(TypeError):
            out  = self.audio_text_integrator.apply(self.filepath, 'k')

        out  = self.audio_text_integrator.apply(self.filepath, self.text_audio_transcription['time'])
        assert pytest.approx(out['text_token_id'], nan_ok=True) == self.text_audio_transcription['text_token_id']
        assert [token in ['', 'maak', 'en', 'er', 'groen', 'als'] for token in out['text_token']]
        assert pytest.approx(out['text_token_start'], nan_ok=True) == self.text_audio_transcription['text_token_start']
        assert pytest.approx(out['text_token_end'], nan_ok=True) == self.text_audio_transcription['text_token_end']
        assert pytest.approx(out['text_sent_id'], nan_ok=True) == self.text_audio_transcription['text_sent_id']
        assert out['text_sent_pos'].shape == np.array(self.text_audio_transcription['text_sent_pos']).shape
        assert out['text_sent_neg'].shape == np.array(self.text_audio_transcription['text_sent_neg']).shape
        assert out['text_sent_neu'].shape == np.array(self.text_audio_transcription['text_sent_neu']).shape


    def test_apply_error(self):
        with pytest.raises(TimeStepError):
            self.audio_text_integrator.apply(self.filepath, time=None)
