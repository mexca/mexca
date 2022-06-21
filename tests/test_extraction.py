""" Test facial feature extraction class and methods """

import json
import os
import numpy as np
import pandas as pd
import pytest
import torch
from mexca.video.extraction import FaceExtractor

def make_reproducible(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

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
        make_reproducible(2022)
        features = self.extractor.apply(self.filepath)
        assert pd.DataFrame(features).to_json() == pytest.approx(self.features, rel=1e-3)
