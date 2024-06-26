"""Test facial feature extraction class and methods.
"""

import json
import os
import subprocess

import numpy as np
import pytest
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans, SpectralClustering
from torch.utils.data import DataLoader

from mexca.data import EMPTY_VALUE, VideoAnnotation
from mexca.video.extraction import (
    FaceExtractor,
    NotEnoughFacesError,
    VideoDataset,
)
from mexca.video.mefarg import MEFARG


class TestVideoDataset:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )

    @pytest.fixture
    def video_dataset(self):
        return VideoDataset(self.filepath)

    def test_duration(self, video_dataset):
        assert isinstance(video_dataset.duration, float)

    def test_len(self, video_dataset):
        assert isinstance(len(video_dataset), int)

    def test_getitem(self, video_dataset):
        item = video_dataset[0]
        assert isinstance(item, dict)
        assert isinstance(item["Image"], torch.Tensor)
        assert item["Frame"] == 0
        assert len(item["Image"].shape) == 3

    def test_getitem_slice(self, video_dataset):
        item = video_dataset[0:1]
        assert isinstance(item, dict)
        assert isinstance(item["Image"], torch.Tensor)
        assert item["Frame"].shape[0] == 1
        assert len(item["Image"].shape) == 4


class TestFaceExtractor:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.mp4"
    )
    with open(
        os.path.join(
            "tests",
            "reference_files",
            "test_video_audio_5_seconds_video_annotation.json",
        ),
        "r",
        encoding="utf-8",
    ) as file:
        features = json.loads(file.read())
    dataset = VideoDataset(filepath, skip_frames=5)
    data_loader = DataLoader(dataset, batch_size=5)

    @pytest.fixture
    def extractor(self):
        return FaceExtractor(num_faces=4)

    @pytest.fixture
    def extractor_min_size(self):
        return FaceExtractor(num_faces=2, post_min_face_size=(0.0, 0.0))

    @staticmethod
    def check_private_attrs(extractor):
        assert not extractor._detector
        assert not extractor._encoder
        assert not extractor._clusterer
        assert not extractor._extractor

    def test_lazy_init(self, extractor):
        self.check_private_attrs(extractor)

        assert isinstance(extractor.detector, MTCNN)
        assert isinstance(extractor.encoder, InceptionResnetV1)
        assert isinstance(extractor.clusterer, ClusterMixin)
        assert isinstance(extractor.extractor, MEFARG)

        del extractor.detector
        del extractor.encoder
        del extractor.clusterer
        del extractor.extractor

        # Check again after deletion
        self.check_private_attrs(extractor)

    def test_detect_face(self, extractor):
        _, boxes, probs, lmks = extractor.detect(self.dataset[5]["Image"])
        assert boxes.shape == (1, 1, 4)
        assert probs.shape == (1, 1)
        assert lmks.shape == (1, 1, 5, 2)

    def test_detect_face_batch(self, extractor):
        _, boxes, probs, lmks = extractor.detect(self.dataset[5:6]["Image"])
        assert boxes.shape == (1, 1, 4)
        assert probs.shape == (1, 1)
        assert lmks.shape == (1, 1, 5, 2)

    def test_detect_face_wrong_dim(self, extractor):
        with pytest.raises(ValueError):
            extractor.detect(self.dataset[5]["Image"][1, :, :])

    def test_detect_no_face(self, extractor):
        _, boxes, probs, lmks = extractor.detect(self.dataset[0]["Image"])
        assert boxes == np.array([None])
        assert probs == np.array([None])
        assert lmks == np.array([None])

    def test_encode(self, extractor):
        faces, _, _, _ = extractor.detect(self.dataset[5]["Image"])
        embeddings = extractor.encode(faces[0])
        assert embeddings.shape == (1, 512)

    def test_identify(self, extractor):
        n_samples = 10
        embeddings = np.random.uniform(0, 1, size=(n_samples, 512))
        labels = extractor.identify(embeddings)
        assert labels.shape == (n_samples,)

    def test_identify_spectral(self):
        extractor = FaceExtractor(
            num_faces=None, clusterer=SpectralClustering(n_clusters=4)
        )
        n_samples = 10
        embeddings = np.random.uniform(0, 1, size=(n_samples, 512))
        labels = extractor.identify(embeddings)
        assert labels.shape == (n_samples,)

    def test_identify_kmeans(self):
        extractor = FaceExtractor(
            num_faces=None, clusterer=KMeans(n_clusters=4)
        )
        n_samples = 10
        embeddings = np.random.uniform(0, 1, size=(n_samples, 512))
        labels = extractor.identify(embeddings)
        assert labels.shape == (n_samples,)

    def test_identify_with_nan(self, extractor):
        n_samples = 10
        embeddings = np.random.uniform(0, 1, size=(n_samples, 512))
        embeddings[8, :] = np.nan
        labels = extractor.identify(embeddings)
        assert labels.shape == (n_samples,)

    def test_identify_not_enough_faces(self, extractor):
        n_samples = 10
        embeddings = np.random.uniform(0, 1, size=(n_samples, 512))

        with pytest.raises(NotEnoughFacesError):
            extractor.identify(embeddings[0:3])

        # When embeddings are not valid
        embeddings[:7, :] = np.nan

        with pytest.raises(NotEnoughFacesError):
            extractor.identify(embeddings)

    def test_extract(self, extractor):
        image = self.dataset[5:6]["Image"]
        faces, _, _, _ = extractor.detect(image)
        aus = extractor.extract(faces)
        assert isinstance(aus, np.ndarray)
        assert np.array(aus).shape == (1, 1, 41)

    def test_extract_no_face(self, extractor):
        image = self.dataset[0:1]["Image"]
        faces, _, _, _ = extractor.detect(image)
        aus = extractor.extract(faces)
        assert isinstance(aus, np.ndarray)
        assert np.array(aus) == np.array([None])

    def test_compute_centroids(self, extractor):
        # create two array embeddings
        v1 = np.random.uniform(size=10)
        v2 = np.random.uniform(size=10)

        embeddings = np.vstack([v1, v1, v2, -v2])

        labels = np.asarray([0, 0, 1, 1])
        centroids, cluster_label_mapping = extractor._compute_centroids(
            embeddings, labels
        )
        # test whether we got two unique labels
        assert len(centroids) == 2
        # the centroid of two arrays that are equal (i.e., v1) is equal to both of them
        assert all(centroids[0] == v1)
        # the centroid of two arrays that are opposite, is equal as all elements equal to 0
        assert all(centroids[1] == 0)
        # centroids must be a list
        assert isinstance(centroids, list)
        # cluster_label_mapping must be a dict
        assert isinstance(cluster_label_mapping, dict)

    def test_compute_confidence(self, extractor):
        # create two array embeddings
        v1 = np.random.uniform(low=-1, high=1, size=10)
        v2 = np.random.uniform(low=-1, high=1, size=10)
        v3 = (v2 + 1.0 * v1) / 2.0

        embeddings = np.vstack([v1, v1, v2, v3])

        labels = [0.0, 0.0, 1.0, 1.0]
        confidence = extractor.compute_confidence(embeddings, labels)

        # assert isistance confidence np.array
        assert len(confidence) == len(labels)

        # I expect first two instances to be equal to 1 as they are both V1
        assert np.isclose(confidence[0], 1)
        assert np.isclose(confidence[1], 1)

        # I expect both V2 and V3 to be less than 1, no particular value
        assert confidence[2] < 1.0
        assert confidence[3] < 1.0

        # I expect confidence of V3 to be less than V2, as this is more close to V1 than V2
        assert confidence[3] < confidence[2]

    def test_calc_face_size(self, extractor):
        bbox = np.array([20, 60, 80, 120])
        size = extractor._calc_face_size(bbox)
        assert size[0] == bbox[2] - bbox[0] and size[1] == bbox[3] - bbox[1]

        with pytest.raises(RuntimeError):
            extractor._calc_face_size(np.flip(bbox))

    def test_apply(self, extractor):
        features = extractor.apply(
            self.filepath, batch_size=5, skip_frames=5, show_progress=False
        )
        assert isinstance(features, VideoAnnotation)

        assert features.frame == self.features["frame"]
        assert features.time == self.features["time"]

        for attr in [
            "frame",
            "time",
            "face_box",
            "face_prob",
            "face_label",
            "face_landmarks",
            "face_aus",
        ]:
            assert len(getattr(features, attr)) == len(self.features[attr])
            assert len(getattr(features, attr)) == len(features.frame)

    def test_apply_no_face_batch(self, extractor_min_size):
        features = extractor_min_size.apply(
            self.filepath,
            batch_size=1,
            skip_frames=5,
            process_subclip=(0, 2),
            show_progress=False,
        )
        assert isinstance(features, VideoAnnotation)
        assert features.face_prob[0] is EMPTY_VALUE

    def test_apply_no_clustering(self, extractor):
        features = extractor.apply(
            self.filepath,
            batch_size=5,
            skip_frames=5,
            cluster_embeddings=False,
            show_progress=False,
        )
        assert isinstance(features, VideoAnnotation)

        assert features.frame == self.features["frame"]
        assert features.time == self.features["time"]

        for attr in [
            "frame",
            "time",
            "face_box",
            "face_prob",
            "face_landmarks",
            "face_aus",
        ]:
            assert len(getattr(features, attr)) == len(self.features[attr])
            assert len(getattr(features, attr)) == len(features.frame)

    def test_apply_return_embeddings(self, extractor):
        features = extractor.apply(
            self.filepath,
            batch_size=5,
            skip_frames=5,
            return_embeddings=True,
            show_progress=False,
        )
        assert isinstance(features, VideoAnnotation)

        assert features.frame == self.features["frame"]
        assert features.time == self.features["time"]

        for attr in [
            "frame",
            "time",
            "face_box",
            "face_prob",
            "face_label",
            "face_landmarks",
            "face_aus",
            "face_embeddings",
        ]:
            assert len(getattr(features, attr)) == len(features.frame)

    def test_cli(self):
        out_filename = (
            os.path.splitext(os.path.basename(self.filepath))[0]
            + "_video_annotation.json"
        )
        subprocess.run(
            [
                "extract-faces",
                "-f",
                self.filepath,
                "-o",
                ".",
                "--num-faces",
                "4",
                "--batch-size",
                "5",
                "--skip-frames",
                "5",
            ],
            check=True,
        )
        assert os.path.exists(out_filename)
        os.remove(out_filename)


class TestFaceExtractorMaxFaces:
    filepath = os.path.join(
        "tests", "test_files", "test_video_multi_5_frames.mp4"
    )

    dataset = VideoDataset(filepath, skip_frames=1)
    data_loader = DataLoader(dataset, batch_size=1)

    @pytest.fixture
    def extractor(self):
        return FaceExtractor(
            num_faces=2, selection_method="num_faces"
        )  # This is actually now the default

    def test_detect_num_faces(self, extractor):
        image = self.dataset[0:5]["Image"]
        faces, boxes, probs, landmarks = extractor.detect(image)

        assert (
            isinstance(boxes, np.ndarray)
            and boxes.shape[1] == extractor.num_faces
        )
        assert (
            isinstance(probs, np.ndarray)
            and probs.shape[1] == extractor.num_faces
        )
        assert (
            isinstance(landmarks, np.ndarray)
            and landmarks.shape[1] == extractor.num_faces
        )
        assert isinstance(faces, list) and np.all(
            [face.shape[0] == extractor.num_faces for face in faces]
        )
