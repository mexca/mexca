""" Test pipeline class and methods """

import json
import os
import platform
import pytest
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline
from mexca.video.extraction import FaceExtractor


class TestPipeline:
    pipeline = Pipeline(
        video=FaceExtractor(  # Only use video because test file contains no audio
            min_clusters=1,
            max_clusters=4
        ),
        audio=None,
        text=None
    )
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_multi_5_frames.mp4'
    )
    with open(os.path.join(
        'tests', 'reference_files', 'features_video_multi_5_frames.json'
    ), 'r', encoding="utf-8") as file:
        features = json.loads(file.read())

    def test_apply(self):
        pipeline_result = self.pipeline.apply(self.filepath)
        assert isinstance(pipeline_result, Multimodal)
        # Only test time feature because the others are covered elsewhere
        assert pipeline_result.features['time'] == self.features['time']

    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='Windows VMs run out of memory when loading entire pipeline')
    def test_from_default(self):
        pipeline = self.pipeline.from_default()
        assert isinstance(pipeline, Pipeline)
