""" Test Audio to text transcription classes and methods """

import json
import os
from pyannote.core import Annotation
from mexca.text.transcription import AudioTranscriber, Sentence, TranscribedSegment


class TestSentence:
    sentence = Sentence('k', 0.0, 1.0)

    def test_set_sentiment(self):
        self.sentence.sent_pos = 0.5
        self.sentence.sent_neg = 0.25
        self.sentence.sent_neu = 0.25

        assert self.sentence.sent_pos == 0.5
        assert self.sentence.sent_neg == 0.25
        assert self.sentence.sent_neu == 0.25


class TestTranscribedSegment:
    segment = TranscribedSegment(0.0, 1.0)

    def test_set_transcription(self):
        self.segment.text = 'k'
        self.segment.lang = 'en'
        self.segment.sents = [Sentence('k', 0.1, 0.9)]

        assert self.segment.text == 'k'
        assert self.segment.lang == 'en'
        assert self.segment.sents == [Sentence('k', 0.1, 0.9)]


class TestAudioTranscriptionEnglish:
    audio_transcriber = AudioTranscriber(whisper_model='tiny')
    filepath = os.path.join(
        'tests', 'test_files', 'test_eng_5_seconds.wav'
    )
    with open(os.path.join(
            'tests', 'reference_files', 'annotation_eng_5_seconds.json'
        ), 'r', encoding="utf-8") as file:
        annotation = Annotation.from_json(json.loads(file.read()))

    ref_text = "I'm carpered down there and I think I think Senator Cooze is there."
    ref_lang = 'en'

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath, self.annotation)
        segments = [seg for seg in transcription.itersegments()]

        # Only one segment
        assert segments[0].text == self.ref_text
        assert segments[0].lang == self.ref_lang
        assert segments[0].sents[0].text == self.ref_text
