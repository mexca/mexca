import os
import pytest
from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.data import Multimodal
from mexca.pipeline import Pipeline
from mexca.text import AudioTranscriber, SentimentExtractor
from mexca.video import FaceExtractor


class TestPipeline:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
    )

    @pytest.fixture
    def full_pipeline(self, num_faces=2, num_speakers=2):
        pipeline = Pipeline(
            face_extractor=FaceExtractor(num_faces=num_faces),
            speaker_identifier=SpeakerIdentifier(
                num_speakers=num_speakers,
                use_auth_token=self.use_auth_token
            ),
            voice_extractor=VoiceExtractor(),
            audio_transcriber=AudioTranscriber(),
            sentiment_extractor=SentimentExtractor()
        )

        return pipeline


    def test_full_pipeline(self, full_pipeline):
        result = full_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5
        )

        assert isinstance(result, Multimodal)

        assert hasattr(result, 'video_annotation')
        assert hasattr(result, 'audio_annotation')
        assert hasattr(result, 'voice_features')
        assert hasattr(result, 'transcription')
        assert hasattr(result, 'sentiment')


