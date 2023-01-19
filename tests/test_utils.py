"""Test utility functions.
"""

import os
import pytest
from mexca.data import AudioTranscription, Multimodal, SentimentAnnotation, SpeakerAnnotation, VideoAnnotation, VoiceFeatures
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


def test_validate_multimodal():
    ref_dir = os.path.join('tests', 'reference_files')
    filepath = 'test_video_audio_5_seconds.mp4'
    multimodal = Multimodal(
        filename=filepath,
        duration=5.0,
        fps=25,
        fps_adjusted=5,
        video_annotation=VideoAnnotation.from_json(
            os.path.join(ref_dir, 'test_video_audio_5_seconds_video_annotation.json')
        ),
        audio_annotation=SpeakerAnnotation.from_rttm(
            os.path.join(ref_dir, 'test_video_audio_5_seconds_audio_annotation.rttm')
        ),
        voice_features=VoiceFeatures.from_json(
            os.path.join(ref_dir, 'test_video_audio_5_seconds_voice_features.json')
        ),
        transcription=AudioTranscription.from_srt(
            os.path.join(ref_dir, 'test_video_audio_5_seconds_transcription.srt')
        ),
        sentiment=SentimentAnnotation.from_json(
            os.path.join(ref_dir, 'test_video_audio_5_seconds_sentiment.json')
        )
    )

    multimodal.merge_features()

    _validate_multimodal(multimodal)
