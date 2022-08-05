""" Test Audio to text transcription classes and methods """

import json
import os
import pytest
from mexca.core.exceptions import ModelTranscriberInitError
from mexca.text.transcription import AudioTranscriber


class TestAudioTranscriptionDutch:
    audio_transcriber = AudioTranscriber(language='dutch')
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    with open(
        os.path.join('tests', 'reference_files', 'transcription_dutch_5_seconds.json'),
        'r', encoding="utf-8") as file:
        ref_transcription = json.loads(file.read())

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath)
        assert all(token in ['', 'maak', 'en', 'er', 'groen', 'als']
                   for token in transcription['transcription'].split(' '))
        assert len(transcription['start_timestamps']) == len(self.ref_transcription['start_timestamps'])
        assert len(transcription['end_timestamps']) == len(self.ref_transcription['end_timestamps'])
        # Large difference between probabilities across different os
        # assert pytest.approx(transcription['probabilities'], rel = 1e-2) == self.ref_transcription['probabilities']


    def test_model_transcriber_init_error(self):
        with pytest.raises(ModelTranscriberInitError):
            AudioTranscriber(language=None)


class TestAudioTranscriptionEnglish:
    audio_transcriber = AudioTranscriber(language='english')
    filepath = os.path.join('tests', 'test_files', 'test_eng_1_second.wav')

    with open(
        os.path.join('tests', 'reference_files', 'transcription_eng_1_second.json'),
        'r', encoding="utf-8") as file:
        ref_transcription = json.loads(file.read())

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath)
        assert all(token in ['senator', 'top', 'corp']
                   for token in transcription['transcription'].split(' '))
        assert len(transcription['start_timestamps']) == len(self.ref_transcription['start_timestamps'])
        assert len(transcription['end_timestamps']) == len(self.ref_transcription['end_timestamps'])
        # Large difference between probabilities across different os
        # assert pytest.approx(transcription['probabilities'], rel = 1e-2) == self.ref_transcription['probabilities']
