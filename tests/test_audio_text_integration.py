""" Test Audio text integration classes and methods """

import os
import json
from mexca.text.transcription import AudioTextIntegrator


class TestAudioTextIntegration:
    audio_text_integrator = AudioTextIntegrator(language='dutch')
    audio_filepath = os.path.join('tests', 'audio_files', 'test_dutch_5_seconds.wav')

    # reference output
    with open(os.path.join(
            'tests', 'reference_files', 'text_audio_integration.json'), 'r') as file:
        text_audio_transcription = json.loads(file.read())

    def test_apply(self):

        out  = self.audio_text_integrator.apply(self.audio_filepath)
        assert out['speech_start'] == self.text_audio_transcription['speech_start']
        assert out['speech_end'] == self.text_audio_transcription['speech_end']
        assert out['speaker'] == self.text_audio_transcription['speaker']
        assert out['text'] == self.text_audio_transcription['text']


