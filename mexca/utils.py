"""Utility functions.
"""

import numpy as np
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
        self.message = "Initialized class instance"

    def __str__(self):
        return self.message


def _validate_face_features(multimodal: Multimodal):
    assert multimodal.features.face_box.dtype == "object"
    assert multimodal.features.face_prob.dtype == "float64"
    assert multimodal.features.face_landmarks.dtype == "object"
    assert multimodal.features.face_aus.dtype == "object"
    assert multimodal.features.face_label.dtype == "float64"
    assert multimodal.features.face_confidence.dtype == "float64"

    assert all(len(bbox) == 4 for bbox in multimodal.features.face_box.dropna())
    assert all(len(lmks) == 68 for lmks in multimodal.features.face_landmarks.dropna())
    assert all(len(aus) == 20 for aus in multimodal.features.face_aus.dropna())

    assert (
        multimodal.features.face_box.isna()
        .eq(multimodal.features.face_prob.isna())
        .all()
    )
    assert (
        multimodal.features.face_box.isna()
        .eq(multimodal.features.face_landmarks.isna())
        .all()
    )
    assert (
        multimodal.features.face_box.isna()
        .eq(multimodal.features.face_aus.isna())
        .all()
    )
    assert (
        multimodal.features.face_box.isna()
        .eq(multimodal.features.face_label.isna())
        .all()
    )


def _validate_speech_segments(multimodal: Multimodal):
    assert multimodal.features.segment_start.dtype == "float64"
    assert multimodal.features.segment_end.dtype == "float64"
    assert multimodal.features.segment_speaker_label.dtype == "object"
    assert multimodal.features.segment_start.le(
        multimodal.features.time, fill_value=0
    ).all()
    assert multimodal.features.segment_end.ge(
        multimodal.features.time, fill_value=multimodal.features.time.max()
    ).all()
    assert (
        multimodal.features.segment_start.dropna()
        .lt(multimodal.features.segment_end.dropna())
        .all()
    )
    assert (
        multimodal.features.segment_start.isna()
        .eq(multimodal.features.segment_end.isna())
        .all()
    )
    assert (
        multimodal.features.segment_start.isna()
        .eq(multimodal.features.segment_speaker_label.isna())
        .all()
    )

    for seg in multimodal.audio_annotation.items():
        assert seg.begin in multimodal.features.segment_start.to_numpy()
        assert seg.end in multimodal.features.segment_end.to_numpy()
        assert str(
            seg.data.name
        ) in multimodal.features.segment_speaker_label.to_numpy().astype(str)


def _validate_voice_feature(
    feat: pd.Series, ref_feat: np.ndarray, d_type: str = "float64", is_pos: bool = False
):
    assert feat.dtype == d_type
    assert len(feat.dropna()) > 0
    if is_pos:
        assert feat[np.isfinite(feat)] > 0

    for f in feat[:-1]:
        if np.isfinite(f):
            assert f in ref_feat


def _validate_voice_features(multimodal: Multimodal):
    for feat_name in multimodal.voice_features.__dict__:
        if feat_name not in (
            "frame",
            "time",
            "hnr_db",
            "f1_amplitude_rel_f0",
            "f2_amplitude_rel_f0",
            "f3_amplitude_rel_f0",
            "h1_f3_diff_db",
        ):
            _validate_voice_feature(
                multimodal.features[feat_name],
                getattr(multimodal.voice_features, feat_name),
            )


def _validate_transcription(multimodal: Multimodal):
    assert multimodal.features.span_start.dtype == "float64"
    assert multimodal.features.span_end.dtype == "float64"
    assert multimodal.features.span_text.dtype == "object"
    assert multimodal.features.span_start.le(
        multimodal.features.time, fill_value=0
    ).all()
    assert multimodal.features.span_end.ge(
        multimodal.features.time, fill_value=multimodal.features.time.max()
    ).all()
    # assert multimodal.features.span_start.le(multimodal.features.segment_end, fill_value=0).all()
    # assert multimodal.features.span_end.le(multimodal.features.segment_end, fill_value=0).all()
    assert (
        multimodal.features.span_start.isna()
        .eq(multimodal.features.span_end.isna())
        .all()
    )
    assert (
        multimodal.features.span_start.isna()
        .eq(multimodal.features.span_text.isna())
        .all()
    )

    for seg in multimodal.transcription.subtitles.items():
        assert seg.begin in multimodal.features.span_start.to_numpy()
        assert seg.end in multimodal.features.span_end.to_numpy()
        assert seg.data.text in multimodal.features.span_text.to_numpy()


def _validate_sentiment(multimodal: Multimodal):
    assert multimodal.features.span_sent_pos.dtype == "float64"
    assert multimodal.features.span_sent_neg.dtype == "float64"
    assert multimodal.features.span_sent_neu.dtype == "float64"
    assert (
        multimodal.features.span_start.isna()
        .eq(multimodal.features.span_sent_pos.isna())
        .all()
    )
    assert (
        multimodal.features.span_start.isna()
        .eq(multimodal.features.span_sent_neg.isna())
        .all()
    )
    assert (
        multimodal.features.span_start.isna()
        .eq(multimodal.features.span_sent_neu.isna())
        .all()
    )

    for seg in multimodal.sentiment.items():
        assert seg.begin in multimodal.features.span_start.to_numpy()
        assert seg.end in multimodal.features.span_end.to_numpy()
        assert seg.data.pos in multimodal.features.span_sent_pos.to_numpy()
        assert seg.data.neg in multimodal.features.span_sent_neg.to_numpy()
        assert seg.data.neu in multimodal.features.span_sent_neu.to_numpy()


def _validate_multimodal(
    output: Multimodal,
    check_video_annotation: bool = True,
    check_audio_annotation: bool = True,
    check_voice_features: bool = True,
    check_transcription: bool = True,
    check_sentiment: bool = True,
):
    assert isinstance(output, Multimodal)

    if check_video_annotation:
        assert isinstance(output.video_annotation, VideoAnnotation)
        _validate_face_features(output)
    if check_audio_annotation:
        assert isinstance(output.audio_annotation, SpeakerAnnotation)
        _validate_speech_segments(output)
    if check_voice_features:
        assert isinstance(output.voice_features, VoiceFeatures)
        _validate_voice_features(output)
    if check_transcription:
        assert isinstance(output.transcription, AudioTranscription)
        _validate_transcription(output)
    if check_sentiment:
        assert isinstance(output.sentiment, SentimentAnnotation)
        _validate_sentiment(output)

    assert isinstance(output.features, pd.DataFrame)

    assert output.features.frame.le(125).all() and output.features.frame.ge(0).all()
    assert output.features.time.le(5.0).all() and output.features.time.ge(0.0).all()
