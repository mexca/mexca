""" Test pipeline class and methods """

import os
from mexca.video.extraction import FaceExtractor
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline

class TestPipeline:
    pipeline = Pipeline(
        video=FaceExtractor(
            min_clusters=1,
            max_clusters=4
        )
    )
    filepath = filepath = os.path.join(
        'tests', 'video_files', 'test_video_multi_5_frames.mp4'
    )

    def test_apply(self):
        pipeline_result = self.pipeline.apply(self.filepath)
        assert isinstance(pipeline_result, Multimodal)
