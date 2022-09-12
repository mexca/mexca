""" Test pipeline class and methods """

import os
import platform
import pytest
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber
from mexca.video.extraction import FaceExtractor


class TestPipeline:
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
    )

    @pytest.mark.skip(reason="""VMs run out of memory when running pipeline on example.
    We cannot choose a smaller example because pipeline requires sufficient frames.""")
    def test_apply(self):
        pipeline = Pipeline(
            video=FaceExtractor(min_clusters=1, max_clusters=3),
            audio=AudioIntegrator(
                SpeakerIdentifier(),
                VoiceExtractor()
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(language='english')
            )
        )
        pipeline_result = pipeline.apply(
            self.filepath,
            show_video_progress=False,
            show_audio_progress=False
        )
        # Only test if pipeline completes because the features are covered elsewhere
        assert isinstance(pipeline_result, Multimodal)

    @pytest.mark.skip(reason="""VMs run out of memory when initiating default pipeline.""")
    def test_from_default(self):
        pipeline = Pipeline().from_default()
        assert isinstance(pipeline, Pipeline)


    @pytest.mark.skipif(
        platform.system() == 'Windows',
        reason='VMs run out of memory on windows'
    )
    def test_pipeline_video(self):
        pipeline_video = Pipeline(
            video=FaceExtractor(min_clusters=1, max_clusters=3)
        )
        pipeline_result = pipeline_video.apply(
            self.filepath,
            show_video_progress=False,
            show_audio_progress=False
        )
        assert isinstance(pipeline_result, Multimodal)


    @pytest.mark.skipif(
        platform.system() == 'Windows',
        reason='VMs run out of memory on windows'
    )
    def test_pipeline_audio(self):
        pipeline_audio = Pipeline(
            audio=AudioIntegrator(
                SpeakerIdentifier(num_speakers=2),
                VoiceExtractor(time_step=0.08)
            )
        )
        pipeline_result = pipeline_audio.apply(
            self.filepath,
            show_video_progress=False,
            show_audio_progress=False
        )
        assert isinstance(pipeline_result, Multimodal)


    @pytest.mark.skipif(
        platform.system() == 'Windows',
        reason='VMs run out of memory on windows'
    )
    def test_pipeline_text(self):
        pipeline_text = Pipeline(
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(language='english'),
                time_step=0.08
            )
        )
        pipeline_result = pipeline_text.apply(
            self.filepath,
            show_video_progress=False,
            show_audio_progress=False
        )
        assert isinstance(pipeline_result, Multimodal)
