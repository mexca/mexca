"""Utility functions.
"""

import pandas as pd
from mexca.data import (AudioTranscription, Multimodal, SentimentAnnotation, SpeakerAnnotation, VideoAnnotation,
                        VoiceFeatures)


# Adapted from whisper.utils
# See: https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/utils.py

def str2bool(string: str):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]

    raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string: str):
    return None if string == "None" else int(string)


def optional_float(string: str):
    return None if string == "None" else float(string)


def optional_str(string: str):
    return None if string == "None" else str(string)


def bool_or_str(string: str):
    try: 
        return str2bool(string)
    except ValueError:
        return string


class ClassInitMessage:
    def __init__(self):
        self.message = 'Initialized class instance'

    def __str__(self):
        return self.message


def _validate_speech_segments(features: pd.DataFrame):
    assert features.segment_start.le(features.time, fill_value=0).all()
    assert features.segment_end.ge(features.time, fill_value=features.time.max()).all()
    assert features.segment_start.dropna().lt(features.segment_end.dropna()).all()
    assert features.segment_start.isna().eq(features.segment_end.isna()).all()
    assert features.segment_start.isna().eq(features.segment_speaker_label.isna()).all()


def _validate_transcription(features: pd.DataFrame):
    assert features.span_start.le(features.time, fill_value=0).all()
    assert features.span_end.ge(features.time, fill_value=features.time.max()).all()
    # assert features.span_start.le(features.segment_end, fill_value=0).all()
    # assert features.span_end.le(features.segment_end, fill_value=0).all()
    assert features.span_start.isna().eq(features.span_end.isna()).all()
    assert features.span_start.isna().eq(features.span_text.isna()).all()


def _validate_sentiment(features: pd.DataFrame):
    assert features.span_start.isna().eq(features.span_sent_pos.isna()).all()
    assert features.span_start.isna().eq(features.span_sent_neg.isna()).all()
    assert features.span_start.isna().eq(features.span_sent_neu.isna()).all()


def _validate_multimodal(
        output: Multimodal,
        check_video_annotation: bool = True,
        check_audio_annotation: bool = True,
        check_voice_features: bool = True,
        check_transcription: bool = True,
        check_sentiment: bool = True
    ):
    assert isinstance(output, Multimodal)

    if check_video_annotation:
        assert isinstance(output.video_annotation, VideoAnnotation)
    if check_audio_annotation:
        assert isinstance(output.audio_annotation, SpeakerAnnotation)
        _validate_speech_segments(output.features)
    if check_voice_features:
        assert isinstance(output.voice_features, VoiceFeatures)
    if check_transcription:
        assert isinstance(output.transcription, AudioTranscription)
        _validate_transcription(output.features)
    if check_sentiment:
        assert isinstance(output.sentiment, SentimentAnnotation)
        _validate_sentiment(output.features)

    assert isinstance(output.features, pd.DataFrame)

    assert output.features.frame.le(125).all() and output.features.frame.ge(0).all()
    assert output.features.time.le(5.0).all() and output.features.time.ge(0.0).all()
