"""Detect and identify faces in a video file.
Extract facial features such as landmarks and action units.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import feat
import numpy as np
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
from spectralcluster import SpectralClusterer
from tqdm import tqdm
from mexca.core.exceptions import SkipFramesError


class FaceExtractor:
    """Combine steps to extract features from faces in a video file.

    Parameters
    ----------
    au_model: str, default='JAANET'
        The name of the pretrained model for detecting action units. Currently, only `'JAANET'` is available.
    landmark_model: {'PFLD', 'MobileFaceNet', 'MobileNet'}
        The name of the pretrained model for detecting facial landmarks. Default is `PFLD`.
    **clargs: dict, optional
        Additional arguments that are passed to the ``spectralcluster.SpectralClusterer`` class instance.

    Attributes
    ----------
    mtcnn
        The MTCNN model for face detection and extraction.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ for details.
    resnet
        The ResnetV1 model for computing face embeddings.
        Uses the pretrained 'vggface2' version by default.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ for details.
    cluster
        The spectral clustering model for identifying faces based on embeddings.
        See `spectralcluster <https://wq2012.github.io/SpectralCluster/>`_ for details.
    pyfeat
        The model for extracting facial landmarks and action units.
        See `py-feat <https://py-feat.org/pages/api.html>`_ for details.



    Notes
    -----
    For details on the available `au_model` and `landmark_model` arguments,
    see the documentation of `py-feat <https://py-feat.org/pages/models.html>`_.
    The pretrained action unit models return different outputs: `JAANET` returns intensities (0-1) for 12 action units,
    whereas `svm` and `logistic` return presence/absence (1/0) values for 20 action units.

    """


    def __init__(self,
        au_model: str = 'JAANET',
        landmark_model: str = 'PFLD',
        **clargs: Any
    ):
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(
            pretrained='vggface2'
        ).eval()
        self.cluster = SpectralClusterer(**clargs)

        if au_model.lower() != 'jaanet':
            raise ValueError('Only the "JAANET" model is available for AU detection')
        else:
            self.pyfeat = feat.detector.Detector(
                au_model=au_model,
                landmark_model=landmark_model
            )


    def detect(self, frame: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Detect faces in an image array.

        Parameters
        ----------
        frame: numpy.ndarray
            Array containing the RGB values of a video frame with dimensions (H, W, 3).

        Returns
        -------
        faces: torch.Tensor
            Tensor containing the N cropped face images from the frame with dimensions (N, 3, 160, 160).
        boxes: numpy.ndarray
            Array containing the bounding boxes of the N detected faces as (x1, y1, x2, y2) coordinates with
            dimensions (N, 4).
        probs: numpy.ndarray
            Probabilities of the detected faces.

        """

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, probs = self.mtcnn.detect(img, landmarks=False) # pylint: disable=unbalanced-tuple-unpacking

        faces = self.mtcnn.extract(frame, boxes, save_path=None)

        return faces, boxes, probs


    def encode(self, faces: torch.Tensor) -> np.ndarray:
        """Compute embeddings for face images.

        Parameters
        ----------
        faces: torch.Tensor
            Tensor containing N face images with dimensions (N, 3, H, W). H and W must at least be 80 for
            the encoding to work.

        Returns
        -------
        numpy.ndarray
            Array containing embeddings of the N face images with dimensions (N, 512).

        """
        embeddings = self.resnet(faces).numpy()

        return embeddings


    def identify(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster faces based on their embeddings.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Array containing embeddings of the N face images with dimensions (N, E) where E is the length
            of the embedding vector.

        Returns
        -------
        numpy.ndarray
            Cluster indices for the N face embeddings.

        """
        labels = np.full((embeddings.shape[0]), np.nan)
        label_finite = np.all(np.isfinite(embeddings), 1)
        labels[label_finite] = self.cluster.predict(
            embeddings[label_finite, :])

        return labels


    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect facial action units and landmarks.

        Parameters
        ----------
        frame: numpy.ndarray
            Array containing the RGB values of a video frame with dimensions (H, W, 3).
        boxes: numpy.ndarray
            Array containing the bounding boxes of the N detected faces as (x1, y1, x2, y2) coordinates
            with dimensions (N, 4).

        Returns
        -------
        landmarks: numpy.ndarray
            Array containg facial landmarks for N detected faces as (x, y) coordinates with dimensions (N, 68, 2).
        aus: numpy.ndarray
            Array containing action units for N detected faces with dimensions (N, U).
            The number of detected actions units U varies across `au_model` specifications.

        """
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)  # convert to 4d

        boxes_list = boxes.reshape(1, -1, 4).tolist()
        landmarks = self.pyfeat.detect_landmarks(frame, boxes_list)
        if self.pyfeat['au_model'].lower() in ['svm', 'logistic']:
            hog, new_landmarks = self.pyfeat._batch_hog(  # pylint: disable=protected-access
                frames=frame, detected_faces=boxes_list, landmarks=landmarks
            )
            aus = self.pyfeat.detect_aus(hog, new_landmarks)
        else:
            aus = self.pyfeat.detect_aus(frame, landmarks)

        # Remove first redundant dimension from landmarks array; new first dim = # of detected faces
        landmarks_np = np.array(landmarks).reshape((-1, 68, 2))

        return landmarks_np, aus


    @staticmethod
    def compute_centroids(
        embs: np.ndarray,
        labels: List[Union[str, int, float]]
    ) -> Tuple[List[float], Dict[Union[str, int, float], int]]:
        """ Compute embeddings' centroids

        Parameters
        ----------
        embs: numpy.ndarray
            embeddings
        labels_unique: list
            face labels

        Returns
        -------
        centroids: list
            embeddings' centroids
        cluster_label_mapping: dict
            cluster label mappings

        """

        # compute unique cluster labels
        unique_labels = np.unique(labels)
        # get rid of nan label
        unique_labels = unique_labels[np.logical_not(np.isnan(unique_labels))]
        # compute centroids:
        centroids = []
        cluster_label_mapping = {}
        centroid = []

        for i, label in enumerate(unique_labels):
            # extract embeddings that have given label
            label_embeddings = embs[labels == label]
            # compute centroid of label_emeddings (vector mean)
            centroid = np.nanmean(label_embeddings, axis=0)
            # appends centroid list
            centroids.append(centroid)
            cluster_label_mapping[label] = i
        return centroids, cluster_label_mapping


    def compute_confidence(self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """ Compute label classification confidence

        Parameters
        ----------
        embeddings: numpy.ndarray
            Array containing embeddings

        labels: numpy.ndarray
            Array containing labels

        Returns
        -------
        confidence: numpy.ndarray
            Array containing confidence scores

        """
        centroids, cluster_label_mapping = self.compute_centroids(embeddings, labels)

        # create empty array with same lenght as labels
        confidence = np.empty_like(labels)

        for i, (frame_embedding, label) in enumerate(zip(embeddings, labels)):
            # if frame is unclassified, assigns zero confidence
            if np.isnan(label):
                confidence[i] = np.nan
            else:
                # compute distance between frame embedding and each centroid
                # frame embedding is put in list becouse it is required by cosine distances API
                # cosine distances returns a list of list, so [0] is taken to get correct shape
                distances = cosine_distances([frame_embedding], centroids)[0]
                # if len(distance) is <= 1, it means that there is only 1 face
                # Clustering confidence is not useful in that case
                if len(distances) <= 1:
                    c = np.nan
                else:
                    # recover index of centroid of cluster
                    cluster_centroid_idx = cluster_label_mapping[label]

                    # distance to the centroid of the cluster to which the frame belongs
                    d1 = distances[cluster_centroid_idx]
                    # mimimum of all other distance

                    d2 = np.min(distances[np.arange(len(distances)) != cluster_centroid_idx])

                    # confidence score: 0 if d1 = d2, 1 if d1 = 0
                    c = (d2 - d1) / d2

                    # handle edge cases: d1<d2 by definition, but if the clustering that produced
                    # the labels used a different distance definition, this could not be the case.
                    # These edge cases are low in confidence, i.e., c = 0.
                    if c < 0:
                        c = 0

                confidence[i] = c

        return confidence


    def apply(self, # pylint: disable=too-many-locals
        filepath: str,
        skip_frames: int = 1,
        process_subclip: Tuple[Optional[float]] = (0, None),
        show_progress: bool = True
    ) -> Dict[str, List[Union[str, float, int, List, np.ndarray]]]:
        """Apply multiple steps to extract features from faces in a video file.

        This method subsequently calls other methods for each frame of a video file to detect and cluster faces.
        It also extracts the facial landmarks and action units.

        Parameters
        ----------
        filepath: str or path
            Path to the video file.
        skip_frames: int, default=1
            Forces extractor to only process every nth frame.
        process_subclip: tuple, default=(0, None)
            Process only a part of the video clip.
            See `moviepy.editor.VideoFileClip
            <https://moviepy.readthedocs.io/en/latest/ref/VideoClip/VideoClip.html#videofileclip>`_ for details.
        show_progress: bool, default=True
            Enables the display of a progress bar.

        Returns
        -------
        dict
            A dictionary with extracted facial features.

        """
        if skip_frames < 1:
            raise ValueError('Argument "skip_frames" must be >= 1')

        with VideoFileClip(filepath, audio=False, verbose=False) as clip:
            subclip = clip.subclip(process_subclip[0], process_subclip[1])

            features = {
                'frame': [],
                'time': [],
                'face_box': [],
                'face_prob': [],
                'face_landmarks': [],
                'face_aus': []
            }

            embeddings = []  # Embeddings are separate because they won't be returned

            n_frames = int(subclip.duration * subclip.fps)

            if skip_frames > n_frames:
                raise SkipFramesError('Arguments "skip_frames" cannot be higher than the total frames in the video')

            for i, (t, frame) in tqdm(
                enumerate(subclip.iter_frames(with_times=True)),
                total=n_frames,
                disable=not show_progress
            ):
                if i % skip_frames == 0:

                    faces, boxes, probs = self.detect(frame)

                    if faces is None:
                        features['frame'].append(i)
                        features['time'].append(t)
                        features['face_box'].append(np.nan)
                        features['face_prob'].append(np.nan)
                        features['face_landmarks'].append(np.nan)
                        features['face_aus'].append(np.nan)

                        embeddings.append(
                            np.full((self.resnet.last_bn.num_features), np.nan))
                    else:
                        embs = self.encode(faces)  # Embeddings per frame
                        landmarks, aus = self.extract(frame, boxes)

                        for box, prob, emb, landmark, au in zip(boxes, probs, embs, landmarks, aus):
                            features['frame'].append(i)
                            features['time'].append(t)
                            features['face_box'].append(box)
                            features['face_prob'].append(prob)
                            features['face_landmarks'].append(landmark)
                            features['face_aus'].append(au)

                            embeddings.append(emb)

            features['face_id'] = self.identify(
                np.array(embeddings).squeeze()).tolist()

            features['face_id_confidence'] = self.compute_confidence(np.asarray(embeddings), np.asarray(features['face_id'])).tolist()
            return features
