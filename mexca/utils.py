"""Utility functions.
"""

import numpy as np
import polars as pl

from mexca.data import (
    AudioTranscription,
    Multimodal,
    SentimentAnnotation,
    SpeakerAnnotation,
    VideoAnnotation,
    VoiceFeatures,
)

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
    assert multimodal.features.collect().select(
        [
            "filename",
            "frame",
            "time",
            "face_box",
            "face_prob",
            "face_landmarks",
            "face_aus",
            "face_label",
            "face_confidence",
        ]
    ).dtypes == [
        pl.Utf8,
        pl.Int64,
        pl.Float64,
        pl.List(pl.Float64),
        pl.Float64,
        pl.List(pl.List(pl.Float64)),
        pl.List(pl.Float64),
        pl.Utf8,
        pl.Float64,
    ]

    assert (
        multimodal.features.select(
            (pl.col("face_box").drop_nulls().list.len() == 4).all(),
            (pl.col("face_landmarks").drop_nulls().list.len() == 5).all(),
            (pl.col("face_aus").drop_nulls().list.len() == 41).all(),
        )
        .collect()
        .to_series()
        .all()
    )

    assert (
        multimodal.features.select(
            (
                pl.col("face_box").is_null() == pl.col("face_prob").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("face_box").is_null()
                == pl.col("face_landmarks").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (pl.col("face_box").is_null() == pl.col("face_aus").is_null()).all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("face_box").is_null() == pl.col("face_label").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )


def _validate_speech_segments(multimodal: Multimodal):
    assert multimodal.features.collect().select(
        [
            "filename",
            "frame",
            "time",
            "segment_start",
            "segment_end",
            "segment_speaker_label",
        ]
    ).dtypes == [pl.Utf8, pl.Int64, pl.Float64, pl.Float64, pl.Float64, pl.Utf8]
    assert (
        multimodal.features.select(
            (pl.col("segment_start") <= pl.col("time")).drop_nulls().all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (pl.col("segment_end") >= pl.col("time")).drop_nulls().all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("segment_start").is_null()
                == pl.col("segment_end").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )

    for seg in multimodal.audio_annotation.segments.items():
        assert (
            seg.begin
            in multimodal.features.select("segment_start").collect().to_series()
        )
        assert (
            seg.end
            in multimodal.features.select("segment_end").collect().to_series()
        )
        assert (
            str(seg.data.name)
            in multimodal.features.select("segment_speaker_label")
            .collect()
            .to_series()
        )


def _validate_voice_feature(
    feat: pl.Series,
    ref_feat: np.ndarray,
):
    assert feat.dtype in (pl.Float32, pl.Float64)
    assert len(feat) > 0

    assert feat.drop_nulls().is_in(ref_feat).all()


def _validate_voice_features(multimodal: Multimodal):
    for feat_name in multimodal.voice_features.__dict__:
        if feat_name not in (
            "filename",
            "frame",
            "time",
            "hnr_db",
            "f1_amplitude_rel_f0",
            "f2_amplitude_rel_f0",
            "f3_amplitude_rel_f0",
            "h1_f3_diff_db",
        ):
            _validate_voice_feature(
                multimodal.features.select(feat_name).collect().to_series(),
                getattr(multimodal.voice_features, feat_name),
            )


def _validate_transcription(multimodal: Multimodal):
    assert multimodal.features.collect().select(
        [
            "filename",
            "frame",
            "time",
            "span_start",
            "span_end",
            "span_text",
            "span_confidence",
        ]
    ).dtypes == [
        pl.Utf8,
        pl.Int64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.Utf8,
        pl.Float64,
    ]
    assert (
        multimodal.features.select(
            (pl.col("span_start") <= pl.col("time")).drop_nulls().all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (pl.col("span_end") >= pl.col("time")).drop_nulls().all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("span_start").is_null() == pl.col("span_end").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )

    for seg in multimodal.transcription.segments.items():
        assert (
            seg.begin
            in multimodal.features.select("span_start").collect().to_series()
        )
        assert (
            seg.end
            in multimodal.features.select("span_end").collect().to_series()
        )
        assert (
            str(seg.data.text)
            in multimodal.features.select("span_text").collect().to_series()
        )


def _validate_sentiment(multimodal: Multimodal):
    assert multimodal.features.collect().select(
        ["span_sent_pos", "span_sent_neg", "span_sent_neu"]
    ).dtypes == [pl.Float64, pl.Float64, pl.Float64]
    assert (
        multimodal.features.select(
            (
                pl.col("span_start").is_null()
                == pl.col("span_sent_pos").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("span_start").is_null()
                == pl.col("span_sent_neg").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )
    assert (
        multimodal.features.select(
            (
                pl.col("span_start").is_null()
                == pl.col("span_sent_neu").is_null()
            ).all()
        )
        .collect()
        .to_series()
        .all()
    )

    for seg in multimodal.sentiment.segments.items():
        assert (
            seg.begin
            in multimodal.features.select("span_start").collect().to_series()
        )
        assert (
            seg.end
            in multimodal.features.select("span_end").collect().to_series()
        )
        assert (
            seg.data.pos
            in multimodal.features.select("span_sent_pos").collect().to_series()
        )
        assert (
            seg.data.neg
            in multimodal.features.select("span_sent_neg").collect().to_series()
        )
        assert (
            seg.data.neu
            in multimodal.features.select("span_sent_neu").collect().to_series()
        )


def _validate_multimodal(
    output: Multimodal,
    check_video_annotation: bool = True,
    check_audio_annotation: bool = True,
    check_voice_features: bool = True,
    check_transcription: bool = True,
    check_sentiment: bool = True,
):
    assert isinstance(output, Multimodal)

    assert (
        output.features.select(
            (pl.col("frame") >= 0).all() & (pl.col("time") >= 0).all()
        )
        .collect()
        .to_series()
        .all()
    )

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

    assert isinstance(output.features, pl.LazyFrame)

    assert (
        output.features.select(
            (pl.col("frame") >= 0).all(), (pl.col("time") >= 0).all()
        )
        .collect()
        .to_series()
        .all()
    )
