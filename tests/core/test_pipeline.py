""" Test pipeline class and methods """

import os
import pytest
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline
from mexca.text.transcription import AudioTextIntegrator, AudioTranscriber
from mexca.text.sentiment import SentimentExtractor
from mexca.video.extraction import FaceExtractor


class TestPipeline:
    """VMs run out of memory when running pipeline on example.
    We cannot choose a smaller example because pipeline requires sufficient frames.
    """

    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True

    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
    )

    # Skip tests on GitHub actions runner but allow local runs
    @pytest.mark.skip_env('runner')
    def test_apply(self):
        pipeline = Pipeline(
            video=FaceExtractor(min_clusters=1, max_clusters=3),
            audio=AudioIntegrator(
                SpeakerIdentifier(use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True),
                VoiceExtractor()
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(whisper_model='tiny'),
                sentiment_extractor=SentimentExtractor()
            )
        )
        pipeline_result = pipeline.apply(
            self.filepath,
            show_video_progress=False,
            show_audio_progress=False,
            show_text_progress=False
        )
        # Only test if pipeline completes because the features are covered elsewhere
        assert isinstance(pipeline_result, Multimodal)

    @pytest.mark.skip_env('runner')
    def test_from_default(self):
        pipeline = Pipeline().from_default(use_auth_token=self.use_auth_token)
        assert isinstance(pipeline, Pipeline)


    @pytest.mark.skip_env('runner')
    @pytest.mark.skip_os(['Windows'])
    def test_pipeline_video(self):
        pipeline_video = Pipeline(
            video=FaceExtractor(min_clusters=1, max_clusters=3)
        )
        pipeline_result = pipeline_video.apply(
            self.filepath,
            show_video_progress=False
        )
        assert isinstance(pipeline_result, Multimodal)


    @pytest.mark.skip_env('runner')
    @pytest.mark.skip_os(['Windows'])
    def test_pipeline_audio_text(self):
        # Text depends on audio so they must be tested together
        pipeline_audio = Pipeline(
            audio=AudioIntegrator(
                SpeakerIdentifier(num_speakers=2, use_auth_token=self.use_auth_token),
                VoiceExtractor(time_step=0.08)
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(whisper_model='tiny'),
                sentiment_extractor=SentimentExtractor(),
                time_step=0.08
            )
        )
        pipeline_result = pipeline_audio.apply(
            self.filepath,
            show_audio_progress=False,
            show_text_progress=False
        )
        assert isinstance(pipeline_result, Multimodal)
