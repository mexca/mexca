""" Test facial feature extraction class and methods """

import json
import os
import pandas as pd
from mexca.video.extraction import FaceExtractor

class TestFaceExtractor:
    extractor = FaceExtractor(min_clusters=1, max_clusters=4)
    filepath = os.path.join(
        'tests', 'video_files', 'test_video_multi_5_frames.mp4'
    )
    with open(os.path.join(
            'tests', 'reference_files', 'features_video_multi_5_frames.json'
        ), 'r') as file:
        features = json.loads(file.read())

    def test_apply(self):
        features = self.extractor.apply(self.filepath)
        assert pd.DataFrame(features).to_json() == self.features
