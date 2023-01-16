import os
import pytest
import pandas as pd
from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.data import AudioTranscription, Multimodal, SentimentAnnotation, SpeakerAnnotation, VideoAnnotation, VoiceFeatures
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
            audio_transcriber=AudioTranscriber(whisper_model='tiny'),
            sentiment_extractor=SentimentExtractor()
        )

        return pipeline


    def test_full_pipeline(self, full_pipeline):
        result = full_pipeline.apply(
            self.filepath,
            frame_batch_size=5,
            skip_frames=5,
            keep_audiofile=True # Otherwise test audio file is removed
        )

        assert isinstance(result, Multimodal)
        assert isinstance(result.video_annotation, VideoAnnotation)
        assert isinstance(result.audio_annotation, SpeakerAnnotation)
        assert isinstance(result.voice_features, VoiceFeatures)
        assert isinstance(result.transcription, AudioTranscription)
        assert isinstance(result.sentiment, SentimentAnnotation)
        assert isinstance(result.features, pd.DataFrame)

        assert result.features.frame.le(125).all() and result.features.frame.ge(0).all()
        assert result.features.time.le(5.0).all() and result.features.time.ge(0.0).all()

        assert result.features.segment_start.le(result.features.time, fill_value=0).all()
        assert result.features.segment_end.ge(result.features.time, fill_value=result.features.time.max()).all()
        assert result.features.segment_start.dropna().lt(result.features.segment_end.dropna()).all()
        assert result.features.segment_start.isna().eq(result.features.segment_end.isna()).all()
        assert result.features.segment_start.isna().eq(result.features.segment_speaker_label.isna()).all()

        assert result.features.span_start.le(result.features.time, fill_value=0).all()
        assert result.features.span_end.ge(result.features.time, fill_value=result.features.time.max()).all()
        assert result.features.span_start.isna().eq(result.features.span_end.isna()).all()
        assert result.features.span_start.isna().eq(result.features.span_text.isna()).all()
