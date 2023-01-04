""" Test Audio to text transcription classes and methods """

import json
import os
import subprocess
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


class TestAudioTranscription:
    audio_transcriber = AudioTranscriber(whisper_model='tiny')
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )
    annotation_path = os.path.join(
        'tests', 'reference_files', 'annotation_video_audio_5_seconds.json'
    )
    with open(annotation_path, 'r', encoding="utf-8") as file:
        annotation = Annotation.from_json(json.loads(file.read()))

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath, self.annotation)
        segments = list(transcription.itersegments())

        # Only one segment
        assert isinstance(segments[0].text, str)
        assert isinstance(segments[0].lang, str)
        assert isinstance(segments[0].sents[0].text, str)


    def test_cli(self):
        out_filename = os.path.basename(self.filepath) + '.json'
        subprocess.run(['transcribe', '-f', self.filepath,
                        '-a', self.annotation_path, '-o', '.'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)


class TestWhisper:
    model_size = 'tiny'
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )


    @pytest.fixture
    def model(self):
        return whisper.load_model(self.model_size)


    @pytest.fixture
    def stable_model(self):
        return stable_whisper.load_model(self.model_size)


    def test_transcribe(self, model):
        output = model.transcribe(self.filepath, fp16=False)

        # Test entire text of audio and language detection
        assert isinstance(output['text'].strip(), str)
        assert isinstance(output['language'], str)


    def test_word_ts(self, stable_model):
        output = stable_model.transcribe(self.filepath, fp16=False)
        # Test word level timestamps of first segment
        first_segment = output['segments'][0]

        assert 'whole_word_timestamps' in first_segment

        first_segment_ts = output['segments'][0]['whole_word_timestamps']
        # Test first token of first segment
        assert isinstance(first_segment_ts[0]['word'], str)
        assert isinstance(first_segment_ts[0]['timestamp'], float)
