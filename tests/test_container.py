import os
import pytest
from docker.errors import NotFound
from mexca.container import AudioTranscriberContainer, BaseContainer, FaceExtractorContainer, SentimentExtractorContainer, SpeakerIdentifierContainer, VoiceExtractorContainer
from mexca.pipeline import Pipeline
from mexca.utils import _validate_multimodal


@pytest.mark.skip_env('runner')
class TestBaseContainer:
    def test_invalid_image_name(self):
        with pytest.raises(NotFound):
            BaseContainer(image_name='sdfsdf')


    def test_get_latest_tag(self):
        container = VoiceExtractorContainer(get_latest_tag=True)
        assert isinstance(container, VoiceExtractorContainer)
        assert container.image_name == 'mexca/voice-extractor:latest'


@pytest.mark.skip_env('runner')
class TestPipelineContainer:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
    )
    components = ["face-extractor", "speaker-identifier", "voice-extractor", "audio-transcriber", "sentiment-extractor"]


    @pytest.fixture
    def face_extractor_pipeline(self, num_faces=2):
        pipeline = Pipeline(
            face_extractor=FaceExtractorContainer(num_faces=num_faces)
        )

        return pipeline


    @pytest.fixture
    def speaker_identifier_pipeline(self, num_speakers=2):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifierContainer(
                num_speakers=num_speakers,
                use_auth_token=self.use_auth_token
            )
        )

        return pipeline


    @pytest.fixture
    def voice_extractor_pipeline(self):
        pipeline = Pipeline(
            voice_extractor=VoiceExtractorContainer()
        )

        return pipeline


    @pytest.fixture
    def speaker_identifier_transcription_pipeline(self, num_speakers=2):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifierContainer(
                num_speakers=num_speakers,
                use_auth_token=self.use_auth_token
            ),
            audio_transcriber=AudioTranscriberContainer(whisper_model='tiny')
        )

        return pipeline


    @pytest.fixture
    def speaker_identifier_transcription_sentiment_pipeline(self, num_speakers=2):
        pipeline = Pipeline(
            speaker_identifier=SpeakerIdentifierContainer(
                num_speakers=num_speakers,
                use_auth_token=self.use_auth_token
            ),
            audio_transcriber=AudioTranscriberContainer(whisper_model='tiny'),
            sentiment_extractor=SentimentExtractorContainer()
        )

        return pipeline


    def test_face_extractor_pipeline(self, face_extractor_pipeline):
        result = face_extractor_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True
        )

        _validate_multimodal(result,
            check_audio_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False
        )


    def test_speaker_identifier_pipeline(self, speaker_identifier_pipeline):
        result = speaker_identifier_pipeline.apply(
            self.filepath,
            keep_audiofile=True # Otherwise test audio file is removed
        )

        _validate_multimodal(result,
            check_video_annotation=False,
            check_voice_features=False,
            check_transcription=False,
            check_sentiment=False
        )


    def test_voice_extractor_pipeline(self, voice_extractor_pipeline):
        result = voice_extractor_pipeline.apply(
            self.filepath,
            keep_audiofile=True # Otherwise test audio file is removed
        )

        _validate_multimodal(result,
            check_video_annotation=False,
            check_audio_annotation=False,
            check_transcription=False,
            check_sentiment=False
        )


    def test_speaker_identifier_transcription_pipeline(self, speaker_identifier_transcription_pipeline):
        result = speaker_identifier_transcription_pipeline.apply(
            self.filepath,
            keep_audiofile=True # Otherwise test audio file is removed
        )

        _validate_multimodal(result,
            check_video_annotation=False,
            check_voice_features=False,
            check_sentiment=False
        )


    def test_speaker_identifier_transcription_sentiment_pipeline(self, speaker_identifier_transcription_sentiment_pipeline):
        result = speaker_identifier_transcription_sentiment_pipeline.apply(
            self.filepath,
            keep_audiofile=True # Otherwise test audio file is removed
        )

        _validate_multimodal(result,
            check_video_annotation=False,
            check_voice_features=False
        )
        