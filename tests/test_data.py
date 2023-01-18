"""Test data classes and methods.
"""

import datetime
import os
import pytest
import srt
from intervaltree import Interval, IntervalTree
from pyannote.core import Annotation, Segment
from mexca.data import AudioTranscription, Multimodal, SegmentData, Sentiment, SentimentAnnotation, SpeakerAnnotation, TranscriptionData, VideoAnnotation, VoiceFeatures, _get_rttm_header
from mexca.utils import _validate_multimodal


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
        annotation = VoiceFeatures(frame=[0, 1, 2], time=[0, 1, 2])
        annotation.write_json(filename)
        assert os.path.exists(filename)
        annotation = VoiceFeatures.from_json(filename=filename)
        assert isinstance(annotation, VoiceFeatures)
        assert annotation.frame == [0, 1, 2]
        assert annotation.time == [0, 1, 2]
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
        os.remove(filename)


class TestAudioTranscription:
    def test_write_from_srt(self):
        filename = 'test.srt'
        transcription = AudioTranscription(
            filename=filename,
            subtitles=IntervalTree([
                Interval(
                    begin=0,
                    end=1,
                    data=TranscriptionData(
                        index=0,
                        text="Test"
                    )
                )
            ])
        )

        transcription.write_srt(filename=filename)
        assert os.path.exists(filename)

        transcription = AudioTranscription.from_srt(filename=filename)
        assert isinstance(transcription, AudioTranscription)
        for seg in transcription.subtitles.items():
            assert isinstance(seg.data, TranscriptionData)
        
        os.remove(filename)


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


class TestMultimodal:
    ref_dir = os.path.join('tests', 'reference_files')
    filepath = 'test_video_audio_5_seconds.mp4'

    @pytest.fixture
    def multimodal(self) -> Multimodal:
        return Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            video_annotation=VideoAnnotation.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_video_annotation.json')
            ),
            audio_annotation=SpeakerAnnotation.from_rttm(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_audio_annotation.rttm')
            ),
            voice_features=VoiceFeatures.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_voice_features.json')
            ),
            transcription=AudioTranscription.from_srt(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_transcription.srt')
            ),
            sentiment=SentimentAnnotation.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_sentiment.json')
            )
        )


    def test_merge_features(self, multimodal):
        multimodal.merge_features()
        _validate_multimodal(multimodal)


    def test_merge_features_video_annotation(self):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            video_annotation=VideoAnnotation.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_video_annotation.json')
            )
        )

        output.merge_features()
        _validate_multimodal(output,
            check_audio_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False
        )


    def test_merge_features_audio_annotation(self):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=SpeakerAnnotation.from_rttm(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_audio_annotation.rttm')
            )
        )

        output.merge_features()
        _validate_multimodal(output,
            check_video_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False
        )

    
    def test_merge_features_voice_features(self):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            voice_features=VoiceFeatures.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_voice_features.json')
            )
        )

        output.merge_features()
        _validate_multimodal(output,
            check_video_annotation=False,
            check_audio_annotation=False,
            check_transcription=False,
            check_sentiment=False
        )


    def test_merge_features_transcription(self):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=SpeakerAnnotation.from_rttm(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_audio_annotation.rttm')
            ),
            transcription=AudioTranscription.from_srt(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_transcription.srt')
            )
        )

        output.merge_features()
        _validate_multimodal(output,
            check_video_annotation=False,
            check_voice_features=False,
            check_sentiment=False
        )


    def test_merge_features_sentiment(self):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=SpeakerAnnotation.from_rttm(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_audio_annotation.rttm')
            ),
            transcription=AudioTranscription.from_srt(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_transcription.srt')
            ),
            sentiment=SentimentAnnotation.from_json(
                os.path.join(self.ref_dir, 'test_video_audio_5_seconds_sentiment.json')
            )
        )

        output.merge_features()
        _validate_multimodal(output,
            check_video_annotation=False,
            check_voice_features=False
        )
        