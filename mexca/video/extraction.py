"""Detect and identify faces in a video file.
Extract facial features such as landmarks and action units.
"""

import cv2
import feat
import numpy as np
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from moviepy.editor import VideoFileClip
from PIL import Image
from spectralcluster import SpectralClusterer
from tqdm import tqdm
from mexca.core.exceptions import SkipFramesError
from sklearn.metrics.pairwise import cosine_distances

class FaceExtractor:
    """Combine steps to extract features from faces in a video file.

    Parameters
    ----------
    au_model: {'JAANET', 'svm', 'logistic'}
        The name of the pretrained model for detecting action units. Default is `JAANET`.
    landmark_model: {'PFLD', 'MobileFaceNet', 'MobileNet'}
        The name of the pretrained model for detecting facial landmarks. Default is `PFLD`.
    **clargs: dict, optional
        Additional arguments that are passed to the ``spectralcluster.SpectralClusterer`` class instance.

    Attributes
    ----------
    mtcnn
    resnet
    cluster
    pyfeat

    Notes
    -----
    For details on the available `au_model` and `landmark_model` arguments,
    see the documentation of `py-feat <https://py-feat.org/pages/models.html>`_.
    The pretrained action unit models return different outputs: `JAANET` returns intensities (0-1) for 12 action units,
    whereas `svm` and `logistic` return presence/absence (1/0) values for 20 action units.

    """


    def __init__(self, au_model='JAANET', landmark_model='PFLD', **clargs) -> 'FaceExtractor':
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(
            pretrained='vggface2'
        ).eval()
        self.cluster = SpectralClusterer(**clargs)
        self.pyfeat = feat.detector.Detector(
            au_model=au_model,
            landmark_model=landmark_model
        )


    @property
    def mtcnn(self):
        """The MTCNN model for face detection and extraction.
        Must be instance of ``MTCNN`` class.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ for details.
        """
        return self._mtcnn


    @mtcnn.setter
    def mtcnn(self, new_mtcnn):
        if isinstance(new_mtcnn, MTCNN):
            self._mtcnn = new_mtcnn
        else:
            raise TypeError('Can only set "mtcnn" to instances of the "MTCNN" class')


    @property
    def resnet(self):
        """The ResnetV1 model for computing face embeddings. Uses the pretrained 'vggface2' version by default.
        Must be instance of ``InceptionResnetV1`` class.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ for details.
        """
        return self._resnet


    @resnet.setter
    def resnet(self, new_resnet):
        if isinstance(new_resnet, InceptionResnetV1):
            self._resnet = new_resnet
        else:
            raise TypeError('Can only set "resnet" to instances of the "InceptionResnetV1" class')


    @property
    def cluster(self):
        """The spectral clustering model for identifying faces based on embeddings.
        Must be instance of ``SpectralClusterer`` class.
        See `spectralcluster <https://wq2012.github.io/SpectralCluster/>`_ for details.
        """
        return self._cluster


    @cluster.setter
    def cluster(self, new_cluster):
        if isinstance(new_cluster, SpectralClusterer):
            self._cluster = new_cluster
        else:
            raise TypeError('Can only set "cluster" to instances of the "SpectralClusterer" class')


    @property
    def pyfeat(self):
        """The model for extracting facial landmarks and action units. Must be instance of ``Detector`` class.
        See `py-feat <https://py-feat.org/pages/api.html>`_ for details.
        """
        return self._pyfeat


    @pyfeat.setter
    def pyfeat(self, new_pyfeat):
        if isinstance(new_pyfeat, feat.detector.Detector):
            self._pyfeat = new_pyfeat
        else:
            raise TypeError('Can only set "pyfeat" to instances of the "Detector" class')


    def detect(self, frame):
        """Detect faces in an image array.

        Parameters
        ----------
        frame: numpy.ndarray
            Array containing the RGB values of a video frame with dimensions (H, W, 3).

        Returns
        -------
        faces: torch.tensor
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


    def encode(self, faces):
        """Compute embeddings for face images.

        Parameters
        ----------
        faces: torch.tensor
            Tensor containing N face images with dimensions (N, 3, H, W). H and W must at least be 80 for
            the encoding to work.

        Returns
        -------
        numpy.ndarray
            Array containing embeddings of the N face images with dimensions (N, 512).

        """
        embeddings = self.resnet(faces).numpy()

        return embeddings


    def identify(self, embeddings):
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


    def extract(self, frame, boxes):
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
    def check_skip_frames(skip_frames):
        if isinstance(skip_frames, int):
            if skip_frames < 1:
                raise ValueError('Argument "skip_frames" must be >= 1')
        else:
            raise TypeError('Argument "skip_frames" must be int')


    @staticmethod
    def compute_confidence(embeddings,labels):
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
        # compute unique cluster labels
        unique_labels = np.unique(labels)
        # get rid of nan label
        unique_labels = unique_labels[np.logical_not(np.isnan(unique_labels))]

        # compute centroids:
        centroids = [] # list of centroid
        cluster_label_mapping = {} # maps the cluster label to the index of the centroid list

        for i,label in enumerate(unique_labels):
            # extract embeddings that have given label
            label_embeddings = embeddings[labels==label]
            # compute centroid of label_emeddings (vector mean)
            centroid = np.nanmean(label_embeddings,axis=0)
            # appends centroid list
            centroids.append(centroid)
            cluster_label_mapping[label] = i

        # create empty array with same lenght as labels
        confidence = np.empty_like(labels)

        # cycle over frames
        for i,(frame_embedding,label) in enumerate(zip(embeddings,labels)):
            #if frame is unclassified, assigns zero confidence
            if np.isnan(label):
                confidence[i] = np.nan

            else:
                # compute distance between frame embedding and each centroid
                # frame embedding is put in list becouse it is required by cosine distances API
                # cosine distances returns a list of list, so [0] is taken to get correct shape
                distances = cosine_distances([frame_embedding],centroids)[0]

                cluster_centroid_idx = cluster_label_mapping[label] # recover index of centroid of cluster

                # distance to the centroid of the cluster to which the frame belongs

                d1 = distances[cluster_centroid_idx]
                # mimimum of all other distance
                d2 = np.min(distances[np.arange(len(distances))!=cluster_centroid_idx])

                #confidence score: 0 if d1=d2, 1 if d1 = 0
                c = (d2-d1)/d2

                # handle edge case: in principle d1<d2 by definition, but if the clustering that produced
                # the labels used a different distance definition, this could not be the case. This
                # edge cases are low confidence.
                if c<0:
                    c = 0

                confidence[i] = c

        return confidence

    def apply(self, filepath, skip_frames=1, process_subclip=(0, None), show_progress=True):  # pylint: disable=too-many-locals
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
        if not isinstance(show_progress, bool):
            raise TypeError('Argument "show_progress" must be bool')

        self.check_skip_frames(skip_frames)

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

            n_frames = int(subclip.duration*subclip.fps)

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

            features['face_id_confidence'] = self.compute_confidence(np.asarray(embeddings), np.asarray(features['face_id']))
            return features
