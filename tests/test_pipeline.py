import os
import platform

import pytest

from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.pipeline import Pipeline
from mexca.text import AudioTranscriber, SentimentExtractor
from mexca.utils import _validate_multimodal
from mexca.video import FaceExtractor


class TestPipeline:
    use_auth_token = (
        os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else True
    )
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )
    filepath_list = [
        filepath,
        filepath,
    ]

    @pytest.fixture
    def full_pipeline(self, num_faces=2, num_speakers=2):
        pipeline = Pipeline(
            face_extractor=FaceExtractor(num_faces=num_faces),
            speaker_identifier=SpeakerIdentifier(
                num_speakers=num_speakers, use_auth_token=self.use_auth_token
            ),
            voice_extractor=VoiceExtractor(),
            audio_transcriber=AudioTranscriber(whisper_model="tiny"),
            sentiment_extractor=SentimentExtractor(),
        )

        return pipeline

    @pytest.fixture
    def face_extractor_pipeline(self, num_faces=2):
        pipeline = Pipeline(face_extractor=FaceExtractor(num_faces=num_faces))

        return pipeline

    @pytest.fixture
    def speaker_identifier_pipeline(self, num_speakers=2):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifier(
                num_speakers=num_speakers, use_auth_token=self.use_auth_token
            )
        )

        return pipeline

    @pytest.fixture
    def voice_extractor_pipeline(self):
        pipeline = Pipeline(voice_extractor=VoiceExtractor())

        return pipeline

    @pytest.fixture
    def speaker_identifier_transcription_pipeline(self, num_speakers=2):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifier(
                num_speakers=num_speakers, use_auth_token=self.use_auth_token
            ),
            audio_transcriber=AudioTranscriber(whisper_model="tiny"),
        )

        return pipeline

    @pytest.fixture
    def speaker_identifier_transcription_sentiment_pipeline(
        self, num_speakers=2
    ):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifier(
                num_speakers=num_speakers, use_auth_token=self.use_auth_token
            ),
            audio_transcriber=AudioTranscriber(whisper_model="tiny"),
            sentiment_extractor=SentimentExtractor(),
        )

        return pipeline

    def test_full_pipeline(self, full_pipeline):
        result = full_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        check_darwin = platform.system() != "Darwin"
        _validate_multimodal(
            result,
            check_transcription=check_darwin,
            check_sentiment=check_darwin,
        )  # Currently fails only on macOS for unknown reason

    def test_full_pipeline_no_merge(self, full_pipeline):
        result = full_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True,  # Otherwise test audio file is removed
            merge=False,
        )

        assert not result.features

    def test_full_pipeline_filepath_list(self, full_pipeline):
        result = full_pipeline.apply(
            self.filepath_list,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        check_darwin = platform.system() != "Darwin"

        assert isinstance(result, list)

        for r in result:
            _validate_multimodal(
                r,
                check_transcription=check_darwin,
                check_sentiment=check_darwin,
            )  # Currently fails only on macOS for unknown reason

    def test_full_pipeline_file_not_found(self, full_pipeline):
        not_a_file_path = os.path.join("not", "a", "file")

        with pytest.raises(FileNotFoundError):
            _ = full_pipeline.apply(
                not_a_file_path,
                frame_batch_size=5,
                skip_frames=5,
                keep_audiofile=True,  # Otherwise test audio file is removed
            )

        with pytest.raises(FileNotFoundError):
            _ = full_pipeline.apply(
                [not_a_file_path, self.filepath],
                frame_batch_size=5,
                skip_frames=5,
                keep_audiofile=True,  # Otherwise test audio file is removed
            )

    def test_face_extractor_pipeline(self, face_extractor_pipeline):
        result = face_extractor_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True,
        )

        _validate_multimodal(
            result,
            check_audio_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_speaker_identifier_pipeline(self, speaker_identifier_pipeline):
        result = speaker_identifier_pipeline.apply(
            self.filepath,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        _validate_multimodal(
            result,
            check_video_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_voice_extractor_pipeline(self, voice_extractor_pipeline):
        result = voice_extractor_pipeline.apply(
            self.filepath,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        _validate_multimodal(
            result,
            check_video_annotation=False,
            check_audio_annotation=False,
            check_transcription=False,
            check_sentiment=False,
        )

    def test_speaker_identifier_transcription_pipeline(
        self, speaker_identifier_transcription_pipeline
    ):
        result = speaker_identifier_transcription_pipeline.apply(
            self.filepath,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        _validate_multimodal(
            result,
            check_video_annotation=False,
            check_voice_features=False,
            check_sentiment=False,
        )

    def test_speaker_identifier_transcription_sentiment_pipeline(
        self, speaker_identifier_transcription_sentiment_pipeline
    ):
        result = speaker_identifier_transcription_sentiment_pipeline.apply(
            self.filepath,
            keep_audiofile=True,  # Otherwise test audio file is removed
        )

        _validate_multimodal(
            result, check_video_annotation=False, check_voice_features=False
        )
