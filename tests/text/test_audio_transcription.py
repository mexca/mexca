""" Test Audio to text transcription classes and methods """

import os
import json
import pytest
from mexca.text.transcription import AudioTranscriber

class TestAudioTranscription:
    audio_transcriber = AudioTranscriber(language='dutch')
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    with open(os.path.join(
            'tests', 'reference_files', 'transcription_dutch_5_seconds.json'), 'r') as file:
        ref_transcription = json.loads(file.read())

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath)
        assert all(token in ['', 'maak', 'en', 'er', 'groen', 'als'] for token in transcription['transcription'].split(' '))
        assert transcription['start_timestamps'] == self.ref_transcription['start_timestamps']
        assert transcription['end_timestamps'] == self.ref_transcription['end_timestamps']
        assert pytest.approx(transcription['probabilities'], rel = 1e-2) == self.ref_transcription['probabilities']
