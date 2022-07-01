""" Test pipeline class and methods """

import os
import pytest
from mexca.audio.speaker_id import SpeakerIdentifier
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline
from mexca.text.transcription import AudioTextIntegrator, AudioTranscriber
from mexca.video.extraction import FaceExtractor

@pytest.mark.skip('Requires test file with both video and audio')
class TestPipeline:
    pipeline = Pipeline(
        video=FaceExtractor(
            min_clusters=1,
            max_clusters=4
        ),
        audio=SpeakerIdentifier(),
        text=AudioTextIntegrator(
            audio_transcriber=AudioTranscriber('dutch')
        )
    )
    filepath = filepath = os.path.join(
        'tests', 'test_files', 'test_video_multi_5_frames.mp4'
    )

    def test_apply(self):
        pipeline_result = self.pipeline.apply(self.filepath)
        assert isinstance(pipeline_result, Multimodal)
