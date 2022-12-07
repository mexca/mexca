""" Test Audio to text transcription classes and methods """

import json
import os
import pytest
import stable_whisper
import whisper
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
        segments = list(transcription.itersegments())

        # Only one segment
        assert segments[0].text == self.ref_text
        assert segments[0].lang == self.ref_lang
        assert segments[0].sents[0].text == self.ref_text


class TestWhisper:
    model_size = 'tiny'
    filepath = os.path.join(
        'tests', 'test_files', 'test_eng_5_seconds.wav'
    )

    ref_text = "Senator Tom Carpard down there and I think Senator Cooze is there and I think"
    ref_lang = 'en'
    # Reference word level timestamps for first segment
    ref_ts = [{'word': ' Senator', 'timestamp': 0.31999997794628143},
              {'word': ' Tom', 'timestamp': 0.6099999845027924},
              {'word': ' Carpard', 'timestamp': 1.0},
              {'word': ' down', 'timestamp': 1.2699999809265137},
              {'word': ' there', 'timestamp': 1.409999966621399},
              {'word': ' and', 'timestamp': 1.5299999713897705},
              {'word': ' I', 'timestamp': 1.6699999570846558},
              {'word': ' think', 'timestamp': 1.7799999713897705}]


    @pytest.fixture
    def model(self):
        return whisper.load_model(self.model_size)


    @pytest.fixture
    def stable_model(self):
        return stable_whisper.load_model(self.model_size)


    def test_transcribe(self, model):
        output = model.transcribe(self.filepath, fp16=False)

        # Test entire text of audio and language detection
        assert output['text'].strip() == self.ref_text
        assert output['language'] == self.ref_lang


    def test_word_ts(self, stable_model):
        output = stable_model.transcribe(self.filepath, fp16=False)
        # Test word level timestamps of first segment
        first_segment = output['segments'][0]

        assert 'whole_word_timestamps' in first_segment

        first_segment_ts = output['segments'][0]['whole_word_timestamps']
        # Test first token of first segment
        assert isinstance(first_segment_ts[0]['word'], str)
        assert isinstance(first_segment_ts[0]['timestamp'], float)
