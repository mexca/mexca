""" Test face identification classes and methods """

import json
import os
from mexca.video.face_id import FaceIdentifier

class TestFaceIdentifier:
    detector = FaceIdentifier()
    filepath = os.path.join('tests', 'video_files', 'test_video_single_5_frames.mp4')

    with open(os.path.join('tests', 'reference_files', 'faces_single_5_frames.json'), 'r') as file:
        faces = json.load(file)

    def test_apply(self):
        faces = self.detector.apply(self.filepath)
        assert faces == self.faces
