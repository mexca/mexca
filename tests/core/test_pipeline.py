""" Test pipeline class and methods """

import os
# import platform
# import pytest
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber
from mexca.video.extraction import FaceExtractor


class TestPipeline:
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
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
    )

    def test_apply(self):
        pipeline_result = self.pipeline.apply(self.filepath)
        # Only test if pipeline completes because the features are covered elsewhere
        assert isinstance(pipeline_result, Multimodal)

    # @pytest.mark.skipif(platform.system() == 'Windows',
    #                    reason='Windows VMs run out of memory when loading entire pipeline')
    def test_from_default(self):
        pipeline = self.pipeline.from_default()
        assert isinstance(pipeline, Pipeline)
