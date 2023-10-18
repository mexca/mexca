"""Test data classes and methods.
"""

import json
import os
from typing import List

import numpy as np
import pytest

# from intervaltree import Interval, IntervalTree
from pyannote.core import Annotation, Segment
from pydantic import create_model

from mexca.data import (
    AudioTranscription,
    Interval,
    IntervalTree,
    Multimodal,
    SegmentData,
    SentimentAnnotation,
    SentimentData,
    SpeakerAnnotation,
    TranscriptionData,
    VideoAnnotation,
    VoiceFeatures,
    VoiceFeaturesConfig,
    _check_common_length,
    _check_sorted,
    _float_to_str,
    _get_rttm_header,
    _nan_to_none,
)
from mexca.utils import _validate_multimodal


def test_float_to_str():
    assert _float_to_str(3.0) == "3"
    assert _float_to_str("3") == "3"
    assert _float_to_str(3) == "3"
    assert _float_to_str(0.1) == "0"
    assert _float_to_str(None) is None


def test_nan2None():
    assert _nan_to_none(np.nan) is None
    assert _nan_to_none(3.0) == 3.0
    assert _nan_to_none(None) is None

    with pytest.raises(TypeError):
        _nan_to_none("3")


def test_check_sorted():
    list_sorted = [0, 1, 2]
    assert _check_sorted(list_sorted) == list_sorted
    with pytest.raises(ValueError):
        _check_sorted([1, 0, 2])


def test_check_common_length():
    TestClass = create_model(
        "TestClass",
        frame=(List, [0, 1, 2]),
        b=(List, [0, 1, 2]),
        c=(float, 1.0),
    )
    obj = TestClass()
    assert _check_common_length(obj) == obj
    obj.b = [0, 1]
    with pytest.raises(ValueError):
        _check_common_length(obj)


class BaseTest:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )

    @pytest.fixture
    def default_feature_columns(self) -> List[str]:
        return [
            "filename",
            "time",
            "frame",
            "face_box",
            "face_prob",
            "face_landmarks",
            "face_aus",
            "face_label",
            "face_confidence",
            "segment_start",
            "segment_end",
            "segment_speaker_label",
            "span_start",
            "span_end",
            "span_text",
            "span_confidence",
            "span_sent_pos",
            "span_sent_neg",
            "span_sent_neu",
            "pitch_f0_hz",
            "jitter_local_rel_f0",
            "shimmer_local_rel_f0",
            "hnr_db",
            "f1_freq_hz",
            "f1_bandwidth_hz",
            "f1_amplitude_rel_f0",
            "f2_freq_hz",
            "f2_bandwidth_hz",
            "f2_amplitude_rel_f0",
            "f3_freq_hz",
            "f3_bandwidth_hz",
            "f3_amplitude_rel_f0",
            "alpha_ratio_db",
            "hammar_index_db",
            "spectral_slope_0_500",
            "spectral_slope_500_1500",
            "h1_h2_diff_db",
            "h1_f3_diff_db",
            "mfcc_1",
            "mfcc_2",
            "mfcc_3",
            "mfcc_4",
            "spectral_flux",
            "rms_db",
        ]

    @pytest.fixture
    def speaker_annotation(self) -> SpeakerAnnotation:
        return SpeakerAnnotation(
            filename=self.filepath,
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
        )

    @pytest.fixture
    def transcription(self) -> AudioTranscription:
        return AudioTranscription(
            filename=self.filepath,
            segments=IntervalTree(
                [
                    Interval(
                        begin=2.00,
                        end=2.41,
                        data=TranscriptionData(
                            index=0,
                            text="Thank you, honey.",
                            speaker="1",
                            confidence=0.89898,
                        ),
                    ),
                    Interval(
                        begin=4.47,
                        end=4.67,
                        data=TranscriptionData(
                            index=1,
                            text="I, uh...",
                            speaker="1",
                            confidence=0.89898,
                        ),
                    ),
                ]
            ),
        )

    @pytest.fixture
    def sentiment(self) -> SentimentAnnotation:
        return SentimentAnnotation(
            filename=self.filepath,
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
        )


class TestVideoAnnotation:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )
    destpath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.json"
    )
    schema_path = os.path.join(
        "tests", "reference_files", "VideoAnnotation_schema.json"
    )

    @pytest.fixture
    def annotation(self):
        return VideoAnnotation(
            filename=self.filepath,
            frame=[0, 1, 2],
            time=[0.2, 0.4, 0.6],
            face_box=[[0, 1, 2, 3], [0, 1, 2, 3], None],
            face_prob=[0.5, 0.9, None],
            face_landmarks=[[[0, 1]], [[0, 1]], None],
            face_aus=[[0.1, 0.2], [0.1, 0.2], None],
            face_label=[3.0, 4.0, None],
            face_confidence=[0.9, None, None],
            face_average_embeddings={
                "3": np.random.randn(512),
                "4": np.random.randn(512),
            },
        )

    def test_check_sorted_validators(self, annotation):
        with pytest.raises(ValueError):
            annotation.frame = [1, 0, 2]
        with pytest.raises(ValueError):
            annotation.time = [0.6, 0.4, 0.2]

    def test_check_len_validators(self, annotation):
        with pytest.raises(ValueError):
            annotation.face_box = [[0, 1, 2], [0, 1, 2, 3], None]

        with pytest.raises(ValueError):
            annotation.face_landmarks = [[[0, 1, 2]], [[0, 1]], None]

    def test_check_model_validators(self, annotation):
        with pytest.raises(ValueError):
            annotation.face_box = [[0, 1, 2, 3], None, [0, 1, 2, 3]]

        with pytest.raises(ValueError):
            annotation.face_label = ["1", "2", None]

    def test_write_from_json(self, annotation):
        annotation.write_json(self.destpath)
        assert os.path.exists(self.destpath)
        annotation_loaded = VideoAnnotation.from_json(filename=self.destpath)
        assert annotation == annotation_loaded
        os.remove(self.destpath)

    def test_json_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)
        ref_schema = VideoAnnotation.model_json_schema()
        del schema["description"], ref_schema["description"]

        assert ref_schema == schema


class TestVoiceFeaturesConfig:
    filename = "test.yaml"

    schema_path = os.path.join(
        "tests", "reference_files", "VoiceFeaturesConfig_schema.json"
    )

    @pytest.fixture
    def config(self):
        return VoiceFeaturesConfig()

    def test_write_read(self, config):
        config.write_yaml(self.filename)

        assert os.path.exists(self.filename)

        new_config = config.from_yaml(self.filename)

        assert isinstance(new_config, VoiceFeaturesConfig)

        os.remove(self.filename)

    def test_json_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)
        ref_schema = VoiceFeaturesConfig.model_json_schema()
        del schema["description"], ref_schema["description"]

        assert ref_schema == schema


class TestVoiceFeatures:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )
    destpath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.json"
    )

    @pytest.fixture
    def voice_features(self):
        return VoiceFeatures(
            filename=self.filepath, frame=[0, 1, 2], time=[0, 1, 2]
        )

    def test_check_sorted_validators(self, voice_features):
        with pytest.raises(ValueError):
            voice_features.frame = [1, 0, 2]
        with pytest.raises(ValueError):
            voice_features.time = [0.6, 0.4, 0.2]

    def test_check_len_validators(self, voice_features):
        with pytest.raises(ValueError):
            voice_features.add_feature("test", [0, 1])

    def test_add_feature(self, voice_features):
        voice_features.add_feature("test", [0, 1, 2])
        assert "test" in voice_features.model_fields_set

    def test_add_feature_nan(self, voice_features):
        voice_features.add_feature("test", [0, 1, np.nan])
        assert voice_features.test == [0, 1, None]

    def test_write_from_json(self, voice_features):
        voice_features.add_feature("pitch_f0_hz", [320.0, 330.0, np.nan])
        voice_features.write_json(self.destpath)
        assert os.path.exists(self.destpath)
        voice_features_loaded = VoiceFeatures.from_json(filename=self.destpath)
        assert voice_features == voice_features_loaded
        os.remove(self.destpath)


def test_get_rttm_header():
    header = _get_rttm_header()
    assert isinstance(header, list)
    assert len(header) == 9


class TestSpeakerAnnotation(BaseTest):
    destpath_json = os.path.join(
        "tests",
        "test_files",
        "test_video_audio_5_seconds_audio_annotation.json",
    )
    destpath_rttm = os.path.join(
        "tests",
        "test_files",
        "test_video_audio_5_seconds_audio_annotation.rttm",
    )

    @staticmethod
    def check_object(obj):
        assert isinstance(obj, SpeakerAnnotation)
        assert len(obj.segments) == 2
        assert isinstance(list(obj.segments.items())[0].data, SegmentData)

    def test_from_pyannote(self):
        annotation = Annotation(uri=self.filepath)
        annotation[Segment(0, 1), "A"] = "1"
        annotation[Segment(1, 2), "B"] = "2"

        speaker_annotation = SpeakerAnnotation.from_pyannote(
            annotation=annotation
        )

        self.check_object(speaker_annotation)

    def test_write_from_json(self, speaker_annotation):
        speaker_annotation.write_json(self.destpath_json)
        assert os.path.exists(self.destpath_json)
        speaker_annotation_loaded = SpeakerAnnotation.from_json(
            filename=self.destpath_json
        )
        assert speaker_annotation == speaker_annotation_loaded
        os.remove(self.destpath_json)

    def test_write_from_rttm(self, speaker_annotation):
        speaker_annotation.write_rttm(filename=self.destpath_rttm)
        assert os.path.exists(self.destpath_rttm)

        speaker_annotation = SpeakerAnnotation.from_rttm(self.destpath_rttm)
        self.check_object(speaker_annotation)
        os.remove(self.destpath_rttm)


class TestAudioTranscription(BaseTest):
    destpath_json = os.path.join(
        "tests",
        "test_files",
        "test_video_audio_5_seconds_transcription.json",
    )
    destpath_srt = os.path.join(
        "tests",
        "test_files",
        "test_video_audio_5_seconds_transcription.srt",
    )

    def test_write_from_json(self, transcription):
        transcription.write_json(self.destpath_json)
        assert os.path.exists(self.destpath_json)
        transcription_loaded = AudioTranscription.from_json(
            filename=self.destpath_json
        )
        assert transcription == transcription_loaded
        os.remove(self.destpath_json)

    def test_write_from_srt(self, transcription):
        transcription.write_srt(filename=self.destpath_srt)
        assert os.path.exists(self.destpath_srt)
        transcription = AudioTranscription.from_srt(filename=self.destpath_srt)
        assert isinstance(transcription, AudioTranscription)
        for seg in transcription.segments.items():
            assert isinstance(seg.data, TranscriptionData)

        os.remove(self.destpath_srt)


class TestSentimentAnnotation(BaseTest):
    destpath = os.path.join(
        "tests",
        "test_files",
        "test_video_audio_5_seconds_transcription.json",
    )

    def test_write_from_json(self, sentiment):
        sentiment = SentimentAnnotation(
            filename=self.filepath,
            segments=IntervalTree(
                [
                    Interval(
                        begin=0,
                        end=1,
                        data=SentimentData(
                            text="test", pos=0.4, neg=0.4, neu=0.2
                        ),
                    )
                ]
            ),
        )
        sentiment.write_json(self.destpath)
        assert os.path.exists(self.destpath)
        sentiment_loaded = SentimentAnnotation.from_json(filename=self.destpath)
        assert sentiment == sentiment_loaded
        for sent in sentiment.segments.items():
            assert isinstance(sent.data, SentimentData)
        os.remove(self.destpath)


class TestMultimodal(BaseTest):
    ref_dir = os.path.join("tests", "reference_files")

    @pytest.fixture
    def video_annotation(self) -> VideoAnnotation:
        return VideoAnnotation.from_json(
            os.path.join(
                self.ref_dir, "test_video_audio_5_seconds_video_annotation.json"
            )
        )

    @pytest.fixture
    def voice_features(self) -> VoiceFeatures:
        return VoiceFeatures.from_json(
            os.path.join(
                self.ref_dir, "test_video_audio_5_seconds_voice_features.json"
            )
        )

    @pytest.fixture
    def multimodal(
        self,
        video_annotation,
        speaker_annotation,
        voice_features,
        transcription,
        sentiment,
    ) -> Multimodal:
        return Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            video_annotation=video_annotation,
            audio_annotation=speaker_annotation,
            voice_features=voice_features,
            transcription=transcription,
            sentiment=sentiment,
        )

    def test_merge_features(self, multimodal, default_feature_columns):
        multimodal.merge_features()
        _validate_multimodal(multimodal)
        assert multimodal.features.collect().columns == default_feature_columns

    def test_merge_features_video_annotation(self, video_annotation):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            video_annotation=video_annotation,
        )

        output.merge_features()
        _validate_multimodal(
            output,
            check_audio_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_merge_features_audio_annotation(self, speaker_annotation):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=speaker_annotation,
        )

        output.merge_features()
        _validate_multimodal(
            output,
            check_video_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_merge_features_voice_features(self, voice_features):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            voice_features=voice_features,
        )

        output.merge_features()
        _validate_multimodal(
            output,
            check_video_annotation=False,
            check_audio_annotation=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_merge_features_transcription(
        self, speaker_annotation, transcription
    ):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=speaker_annotation,
            transcription=transcription,
        )

        output.merge_features()
        _validate_multimodal(
            output,
            check_video_annotation=False,
            check_voice_features=False,
            check_sentiment=False,
        )

    def test_merge_features_sentiment(
        self, speaker_annotation, transcription, sentiment
    ):
        output = Multimodal(
            filename=self.filepath,
            duration=5.0,
            fps=25,
            fps_adjusted=5,
            audio_annotation=speaker_annotation,
            transcription=transcription,
            sentiment=sentiment,
        )

        output.merge_features()
        _validate_multimodal(
            output, check_video_annotation=False, check_voice_features=False
        )
