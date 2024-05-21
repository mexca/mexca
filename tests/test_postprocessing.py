"""Test postprocessing functions.
"""

import json
import os

import polars as pl
import pytest
from intervaltree import Interval, IntervalTree

from mexca.data import (
    AudioTranscription,
    Multimodal,
    SegmentData,
    SentimentAnnotation,
    SentimentData,
    SpeakerAnnotation,
    TranscriptionData,
    VideoAnnotation,
    VoiceFeatures,
)
from mexca.postprocessing import (
    AU_REF,
    LANDMARKS_REF,
    get_face_speaker_mapping,
    split_list_columns,
    sub_labels,
)


@pytest.fixture
def multimodal():
    multimodal = Multimodal(
        filename=os.path.join(
            "tests", "test_files", "test_video_audio_5_seconds.mp4"
        ),
        duration=5.0,
        fps=25,
        fps_adjusted=5,
        video_annotation=VideoAnnotation.from_json(
            "tests/reference_files/test_video_audio_5_seconds_video_annotation.json"
        ),
        audio_annotation=SpeakerAnnotation(
            filename="tests/reference_files/test_video_audio_5_seconds_video_annotation.json",
            channel=1,
            segments=IntervalTree(
                [
                    Interval(
                        begin=1.92,
                        end=2.92,
                        data=SegmentData(name="1"),
                    ),
                    Interval(
                        begin=3.86,
                        end=4.87,
                        data=SegmentData(name="1"),
                    ),
                ]
            ),
        ),
        voice_features=VoiceFeatures.from_json(
            "tests/reference_files/test_video_audio_5_seconds_voice_features.json"
        ),
        transcription=AudioTranscription(
            filename="tests/reference_files/test_video_audio_5_seconds_voice_features.json",
            segments=IntervalTree(
                [
                    Interval(
                        begin=2.00,
                        end=2.41,
                        data=TranscriptionData(
                            index=0, text="Thank you, honey.", speaker="1"
                        ),
                    ),
                    Interval(
                        begin=4.47,
                        end=4.67,
                        data=TranscriptionData(
                            index=1, text="I, uh...", speaker="1"
                        ),
                    ),
                ]
            ),
        ),
        sentiment=SentimentAnnotation(
            filename="tests/reference_files/test_video_audio_5_seconds_voice_features.json",
            segments=IntervalTree(
                [
                    Interval(
                        begin=2.00,
                        end=2.41,
                        data=SentimentData(
                            text="Thank you, honey.",
                            pos=0.88,
                            neg=0.02,
                            neu=0.1,
                        ),
                    ),
                    Interval(
                        begin=4.47,
                        end=4.67,
                        data=SentimentData(
                            text="I, uh...", pos=0.1, neg=0.37, neu=0.53
                        ),
                    ),
                ]
            ),
        ),
    )

    multimodal.merge_features()

    return multimodal


def test_split_list_columns_lazy(multimodal):
    features = split_list_columns(
        multimodal.features, au_columns=AU_REF, landmark_columns=LANDMARKS_REF
    )

    assert isinstance(features, pl.LazyFrame)
    assert "face_au_1" in features.columns
    assert "face_box_x1" in features.columns
    assert "face_landmarks_x1" in features.columns


def test_split_list_columns_eager(multimodal):
    features = split_list_columns(
        multimodal.features.collect(),
        au_columns=AU_REF,
        landmark_columns=LANDMARKS_REF,
    )

    assert isinstance(features, pl.DataFrame)
    assert "face_au_1" in features.columns
    assert "face_box_x1" in features.columns
    assert "face_landmarks_x1" in features.columns


def test_split_list_columns_custom_columns(multimodal):
    features = split_list_columns(
        multimodal.features, au_columns=[1, 2, 3], landmark_columns=[1, 2]
    )

    assert isinstance(features, pl.LazyFrame)
    assert "face_au_1" in features.columns
    assert "face_au_4" not in features.columns
    assert "face_box_x1" in features.columns
    assert "face_landmarks_x1" in features.columns
    assert "face_landmarks_x3" not in features.columns


def test_get_face_speaker_mapping(multimodal):
    features = multimodal.features.collect()

    mapping = get_face_speaker_mapping(features)

    assert mapping == {"0": "1"}


def test_sub_labels_lazy(multimodal):
    features = sub_labels(multimodal.features, {"0": "1"}, column="face_label")

    assert isinstance(features, pl.LazyFrame)
    assert (
        features.select(pl.col("face_label").drop_nulls().is_in(["1", "-1"]))
        .collect()
        .to_series()
        .all()
    )


def test_sub_labels_eager(multimodal):
    features = sub_labels(
        multimodal.features.collect(), {"0": "1"}, column="face_label"
    )

    assert isinstance(features, pl.DataFrame)
    assert (
        features.select(pl.col("face_label").drop_nulls().is_in(["1", "-1"]))
        .to_series()
        .all()
    )
