""" Create reference files for face id tests """

import json
import os
from mexca.video.face_id import FaceIdentifier

detector = FaceIdentifier()

faces = detector.apply(os.path.join('tests', 'video_files', 'test_video_single_5_frames.mp4'))

with open(os.path.join('tests', 'reference_files', 'faces_single_5_frames.json'), 'w') as file:
    json.dump(faces, file, indent=4)
