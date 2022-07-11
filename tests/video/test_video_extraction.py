""" Test facial feature extraction class and methods """

import json
import os
import numpy as np
from moviepy.editor import VideoFileClip
from mexca.video.extraction import FaceExtractor


class TestFaceExtractor:
    extractor = FaceExtractor(min_clusters=1, max_clusters=4)
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_multi_5_frames.mp4'
    )
    with open(os.path.join(
            'tests', 'reference_files', 'features_video_multi_5_frames.json'
        ), 'r', encoding="utf-8") as file:
        features = json.loads(file.read())


    def test_detect(self):
        with VideoFileClip(self.filepath, audio=False) as clip:
            features = {
                'box': [],
                'prob': []
            }
            for frame in clip.iter_frames():
                _, boxes, probs = self.extractor.detect(frame)
                for box, prob in zip(boxes, probs):
                    features['box'].append(box.tolist())
                    features['prob'].append(prob)

            assert np.array(features['box']).shape == np.array(self.features['box']).shape
            assert features['prob'] == self.features['prob']


    def test_identify(self):
        with VideoFileClip(self.filepath, audio=False) as clip:
            embeddings = []
            for frame in clip.iter_frames():
                faces, _, _ = self.extractor.detect(frame)
                embs = self.extractor.encode(faces).numpy()

                for emb in embs:
                    embeddings.append(emb)

            labels = self.extractor.identify(np.array(embeddings)).tolist()

            assert labels == self.features['label']


    def test_extract(self):
        with VideoFileClip(self.filepath, audio=False) as clip:
            features = {
                'landmarks': [],
                'aus': []
            }
            for frame in clip.iter_frames():
                _, boxes, _ = self.extractor.detect(frame)
                landmarks, aus = self.extractor.extract(frame, boxes)
                landmarks_np = np.array(landmarks).squeeze()
                for landmark, au in zip(landmarks_np, aus):
                    features['landmarks'].append(landmark)
                    features['aus'].append(au)

            assert np.array(features['landmarks']).shape == np.array(self.features['landmarks']).shape
            assert np.array(features['aus']).shape == np.array(self.features['aus']).shape


    def test_apply(self): # Tests JAANET AU model
        features = self.extractor.apply(self.filepath)
        assert features['frame'] == self.features['frame']
        assert features['time'] == self.features['time']
        assert np.array(features['box']).shape == np.array(self.features['box']).shape
        assert features['prob'] == self.features['prob']
        assert features['label'] == self.features['label']
        assert np.array(features['landmarks']).shape == np.array(self.features['landmarks']).shape
        assert np.array(features['aus']).shape == np.array(self.features['aus']).shape


    def test_pyfeat_svm(self): # Tests SVM AU model
        svm_extractor = FaceExtractor(au_model='svm')
        features = svm_extractor.apply(self.filepath)

        assert np.array(features['aus']).shape == np.array(self.features['aus_svm']).shape


    def test_pyfeat_logistic(self): # Tests logistic AU model
        svm_extractor = FaceExtractor(au_model='logistic')
        features = svm_extractor.apply(self.filepath)

        assert np.array(features['aus']).shape == np.array(self.features['aus_logistic']).shape
