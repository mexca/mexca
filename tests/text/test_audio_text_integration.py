""" Test Audio text integration classes and methods """

import json
import os
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber


class TestAudioTextIntegration:
    audio_text_integrator = AudioTextIntegrator(
        audio_transcriber=AudioTranscriber(language='dutch')
    )
    filepath = os.path.join('tests', 'test_files', 'test_dutch_5_seconds.wav')

    # reference output
    with open(
        os.path.join('tests', 'reference_files', 'reference_dutch_5_seconds.json'),
        'r', encoding="utf-8") as file:
        text_audio_transcription = json.loads(file.read())

    def test_apply(self):
        out  = self.audio_text_integrator.apply(self.filepath, self.text_audio_transcription['time'])
        assert all(out['text_token_id'] == self.text_audio_transcription['text_token_id'])
        assert [token in ['', 'maak', 'en', 'er', 'groen', 'als'] for token in out['text_token']]
        assert all(out['text_token_start'] == self.text_audio_transcription['text_token_start'])
        assert all(out['text_token_end'] == self.text_audio_transcription['text_token_end'])
