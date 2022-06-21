""" Test speaker identification classes and methods """

import os
from mexca.audio.speaker_id import SpeakerIdentifier
from pyannote.database.util import load_rttm

class TestSpeakerIdentifier:
    detector = SpeakerIdentifier()
    filepath = os.path.join('tests', 'audio_files', 'test_audio_5_seconds.wav')

    ref_speakers = load_rttm(os.path.join('tests', 'reference_files', 'reference_audio_5_seconds.rttm'))

    def test_apply(self):
        speakers = self.detector.apply(self.filepath)
        assert are_equal(speakers, self.ref_speakers['test_audio_5_seconds'])




def are_equal(annotation1, annotation2):
    equal = True
    segments1 = annotation1.itertracks(yield_label=True)
    segments2 = annotation2.itertracks(yield_label=True)

    for (speech_turn1, _, speaker1), (speech_turn2, _, speaker2) in zip(segments1,segments2):
        if (int(speech_turn1.start) != int(speech_turn2.start)) or (int(speech_turn1.end) != int(speech_turn2.end)) or (f'{speaker1}' != f'{speaker2}'):
            equal = False

        return equal
