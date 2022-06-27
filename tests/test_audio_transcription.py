""" Test Audio to text transcription classes and methods """

import os
from mexca.text.transcription import AudioTranscriber

class TestAudioTranscription:
    audio_transcriber = AudioTranscriber(language='english')
    filepath = os.path.join('tests', 'audio_files', 'test_eng_5_seconds.wav')

    with open(os.path.join('tests', 'reference_files', 'transcription_eng_5_seconds.txt'), 'r') as file:
        ref_transcription = file.read().replace('\n', '')

    def test_apply(self):
        transcription = self.audio_transcriber.apply([self.filepath])
        assert are_equal(transcription, self.ref_transcription)


def are_equal(transcription1, transcription2):

    reference_transcription = transcription1
    predicted_transcription = transcription2

    equal = (reference_transcription == predicted_transcription)

    return equal
