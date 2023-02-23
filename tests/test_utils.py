"""Test utility functions.
"""

import os
import pytest
from intervaltree import Interval, IntervalTree
from mexca.data import AudioTranscription, Multimodal, SegmentData, SentimentAnnotation, SentimentData, SpeakerAnnotation, TranscriptionData, VideoAnnotation, VoiceFeatures
from mexca.utils import str2bool, optional_int, optional_float, optional_str, bool_or_str, _validate_multimodal


def test_str2bool():
    assert str2bool('True')
    assert not str2bool('False')
    with pytest.raises(ValueError):
        assert str2bool('Apple')


def test_optional_int():
    assert optional_int("None") is None
    assert isinstance(optional_int(3), int)


def test_optional_float():
    assert optional_int("None") is None
    assert isinstance(optional_float(3.0), float)


def test_optional_str():
    assert optional_int("None") is None
    assert isinstance(optional_str('a'), str)


def test_bool_or_stri():
    assert bool_or_str("Apple") == "Apple"


@pytest.fixture
def ref_dir():
    return os.path.join('tests', 'reference_files')


@pytest.fixture
def filepath():
    return 'test_video_audio_5_seconds.mp4'


@pytest.fixture
def video_annotation(ref_dir) -> VideoAnnotation:
    return VideoAnnotation.from_json(
        os.path.join(ref_dir, 'test_video_audio_5_seconds_video_annotation.json')
    )


@pytest.fixture
def audio_annotation(filepath) -> SpeakerAnnotation:
    return SpeakerAnnotation([
        Interval(begin=1.92, end=2.92, data=SegmentData(filename=filepath, channel=0, name=0)),
        Interval(begin=3.86, end=4.87, data=SegmentData(filename=filepath, channel=0, name=0))
    ])


@pytest.fixture
def voice_features(ref_dir) -> VoiceFeatures:
    return VoiceFeatures.from_json(
        os.path.join(ref_dir, 'test_video_audio_5_seconds_voice_features.json')
    )


@pytest.fixture
def transcription(filepath) -> AudioTranscription:
    return AudioTranscription(
        filename=filepath,
        subtitles=IntervalTree([
            Interval(begin=2.00, end=2.41, data=TranscriptionData(index=0, text='Thank you, honey.', speaker='0')),
            Interval(begin=4.47, end=4.67, data=TranscriptionData(index=1, text='I, uh...', speaker='0'))
        ])
    )


@pytest.fixture
def sentiment() -> SentimentAnnotation:
    return SentimentAnnotation([
        Interval(begin=2.00, end=2.41, data=SentimentData(text='Thank you, honey.', pos=0.88, neg=0.02, neu=0.1)),
        Interval(begin=4.47, end=4.67, data=SentimentData(text='I, uh...', pos=0.1, neg=0.37, neu=0.53))
    ])


@pytest.fixture
def multimodal(filepath, video_annotation, audio_annotation, voice_features, transcription, sentiment) -> Multimodal:
    return Multimodal(
        filename=filepath,
        duration=5.0,
        fps=25,
        fps_adjusted=5,
        video_annotation=video_annotation,
        audio_annotation=audio_annotation,
        voice_features=voice_features,
        transcription=transcription,
        sentiment=sentiment
    )


def test_validate_multimodal(multimodal):
    multimodal.merge_features()

    _validate_multimodal(multimodal)
