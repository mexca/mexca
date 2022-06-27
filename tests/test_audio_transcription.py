""" Test Audio to text transcription classes and methods """

import os
import json
from mexca.text.transcription import AudioTranscriber

class TestAudioTranscription:
    audio_transcriber = AudioTranscriber(language='dutch')
    filepath = os.path.join('tests', 'audio_files', 'test_dutch_5_seconds.wav')

    with open(os.path.join(
            'tests', 'reference_files', 'transcription_dutch_1_second.json'), 'r') as file:
        ref_transcription = json.loads(file.read())

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath)
        assert are_equal(transcription['transcription'], self.ref_transcription['transcription'])


def are_equal(transcription1, transcription2):

    reference_transcription = transcription1
    predicted_transcription = transcription2

    equal = (reference_transcription == predicted_transcription)

    return equal
