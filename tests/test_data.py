"""Test data classes and methods.
"""

import datetime
import os
import srt
from intervaltree import Interval
from pyannote.core import Annotation, Segment
from mexca.data import AudioTranscription, SegmentData, Sentiment, SentimentAnnotation, SpeakerAnnotation, VideoAnnotation, VoiceFeatures, _get_rttm_header

class TestVideoAnnotation:
    def test_write_from_json(self):
        filename = 'test.json'
        annotation = VideoAnnotation(frame=[0, 1, 2])
        annotation.write_json(filename)
        assert os.path.exists(filename)
        annotation = VideoAnnotation.from_json(filename=filename)
        assert isinstance(annotation, VideoAnnotation)
        assert annotation.frame == [0, 1, 2]
        os.remove(filename)


class TestVoiceFeatures:
    def test_write_from_json(self):
        filename = 'test.json'
        annotation = VoiceFeatures(frame=[0, 1, 2])
        annotation.write_json(filename)
        assert os.path.exists(filename)
        annotation = VoiceFeatures.from_json(filename=filename)
        assert isinstance(annotation, VoiceFeatures)
        assert annotation.frame == [0, 1, 2]
        os.remove(filename)


def test_get_rttm_header():
    header = _get_rttm_header()
    assert isinstance(header, list)
    assert len(header) == 9


class TestSpeakerAnnotation:
    @staticmethod
    def check_object(obj):
        assert isinstance(obj, SpeakerAnnotation)
        assert len(obj) == 1
        assert isinstance(list(obj.items())[0].data, SegmentData)

    def test_from_pyannote(self):
        annotation = Annotation()
        annotation[Segment(0, 1), 'A'] = 'spk_1'

        speaker_annotation = SpeakerAnnotation.from_pyannote(annotation=annotation)
        
        self.check_object(speaker_annotation)


    def test_write_from_rttm(self):
        filename = 'test.rttm'
        speaker_annotation = SpeakerAnnotation(
            [Interval(0.0, 1.0, data=SegmentData(filename=filename, channel=1, name='spk_1'))]
        )

        speaker_annotation.write_rttm(filename=filename)
        assert os.path.exists(filename)

        speaker_annotation = SpeakerAnnotation.from_rttm(filename)
        self.check_object(speaker_annotation)


class TestAudioTranscription:
    def test_write_from_srt(self):
        filename = 'test.srt'
        transcription = AudioTranscription(
            filename=filename,
            subtitles=[srt.Subtitle(
                index=0,
                start=datetime.timedelta(seconds=0),
                end=datetime.timedelta(seconds=1),
                content='Test.'
            )]
        )

        transcription.write_srt(filename=filename)
        assert os.path.exists(filename)

        transcription = AudioTranscription.from_srt(filename=filename)
        assert isinstance(transcription, AudioTranscription)
        assert isinstance(transcription.subtitles[0], srt.Subtitle)


class TestSentimentAnnotation:
    def test_write_from_json(self):
        filename = 'test.json'
        sentiment = SentimentAnnotation(sentiment=[Sentiment(
            index=0,
            pos=0.4,
            neg=0.4,
            neu=0.2
        )])
        sentiment.write_json(filename)
        assert os.path.exists(filename)
        sentiment = SentimentAnnotation.from_json(filename=filename)
        assert isinstance(sentiment, SentimentAnnotation)
        assert isinstance(sentiment.sentiment[0], Sentiment)
        os.remove(filename)
