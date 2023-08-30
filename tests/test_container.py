import os

import pytest
from docker.errors import DockerException, NotFound
from intervaltree import Interval, IntervalTree

from mexca.container import (
    AudioTranscriberContainer,
    BaseContainer,
    FaceExtractorContainer,
    SentimentExtractorContainer,
    SpeakerIdentifierContainer,
    VoiceExtractorContainer,
)
from mexca.data import (
    AudioTranscription,
    SentimentAnnotation,
    SpeakerAnnotation,
    TranscriptionData,
    VideoAnnotation,
    VoiceFeatures,
    VoiceFeaturesConfig,
)


@pytest.mark.skip_os("Darwin")
class TestBaseContainer:
    def test_invalid_image_name(self):
        with pytest.raises(NotFound):
            BaseContainer(image_name="sdfsdf")


@pytest.mark.run_env("face-extractor")
class TestFaceExtractorContainer:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )
    num_faces = 2

    @pytest.fixture
    def face_extractor(self):
        return FaceExtractorContainer(
            num_faces=self.num_faces, get_latest_tag=True
        )

    def test_apply(self, face_extractor):
        result = face_extractor.apply(
            self.filepath, batch_size=5, skip_frames=5
        )
        assert isinstance(result, VideoAnnotation)

    def test_apply_docker_exception(self, face_extractor):
        with pytest.raises(DockerException):
            face_extractor.apply("non/existent/filepath")


@pytest.mark.run_env("speaker-identifier")
class TestSpeakerIdentifierContainer:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )
    num_speakers = 2

    @pytest.fixture
    def speaker_identifier(self):
        return SpeakerIdentifierContainer(
            num_speakers=self.num_speakers,
            use_auth_token=os.environ["HF_TOKEN"],
            get_latest_tag=True,
        )

    def test_apply(self, speaker_identifier):
        result = speaker_identifier.apply(self.filepath)
        assert isinstance(result, SpeakerAnnotation)

    def test_apply_docker_exception(self, speaker_identifier):
        with pytest.raises(DockerException):
            speaker_identifier.apply("non/existent/filepath")


@pytest.mark.run_env("voice-extractor")
class TestVoiceExtractorContainer:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )
    num_faces = 2

    @pytest.fixture
    def voice_extractor(self):
        return VoiceExtractorContainer(get_latest_tag=True)

    @pytest.fixture
    def config(self):
        return VoiceFeaturesConfig(pitch_upper_freq=2000)

    @pytest.fixture
    def voice_extractor_config(self, config):
        return VoiceExtractorContainer(config=config, get_latest_tag=True)

    def test_apply(self, voice_extractor):
        result = voice_extractor.apply(
            self.filepath, time_step=0.2, skip_frames=1
        )
        assert isinstance(result, VoiceFeatures)

    def test_apply_config(self, voice_extractor_config):
        result = voice_extractor_config.apply(
            self.filepath, time_step=0.2, skip_frames=1
        )
        assert isinstance(result, VoiceFeatures)

    def test_apply_docker_exception(self, voice_extractor):
        with pytest.raises(DockerException):
            voice_extractor.apply(
                "non/existent/filepath", time_step=0.2, skip_frames=1
            )


@pytest.mark.run_env("audio-transcriber")
class TestAudioTranscriberContainer:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )
    annotation_path = os.path.join(
        "tests",
        "reference_files",
        "test_video_audio_5_seconds_audio_annotation.json",
    )
    annotation = SpeakerAnnotation.from_json(annotation_path)
    num_speakers = 2

    @pytest.fixture
    def audio_transcriber(self):
        return AudioTranscriberContainer(
            whisper_model="tiny", get_latest_tag=True
        )

    def test_apply(self, audio_transcriber):
        result = audio_transcriber.apply(self.filepath, self.annotation)
        assert isinstance(result, AudioTranscription)


@pytest.mark.run_env("sentiment-extractor")
class TestSentimentExtractorContainer:
    transcription_path = os.path.join(
        "tests",
        "reference_files",
        "test_video_audio_5_seconds_transcription.json",
    )

    @pytest.fixture
    def transcription(self):
        transcription = AudioTranscription(
            filename=self.transcription_path,
            segments=IntervalTree(
                [
                    Interval(
                        begin=0,
                        end=1,
                        data=TranscriptionData(
                            index=0, text="Today was a good day!", speaker="0"
                        ),
                    )
                ]
            ),
        )

        return transcription

    @pytest.fixture
    def sentiment_extractor(self):
        return SentimentExtractorContainer(get_latest_tag=True)

    def test_apply(self, sentiment_extractor, transcription):
        result = sentiment_extractor.apply(transcription)
        assert isinstance(result, SentimentAnnotation)
