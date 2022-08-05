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


class FaceExtractor:
    """Combine steps to extract features from faces in a video file.
    """

    def __init__(self, au_model='JAANET', landmark_model='PFLD', **clargs) -> 'FaceExtractor':
        """Create a class instance for extracting facial features from a video.

        Parameters
        ----------
        au_model: {'JAANET', 'svm', 'logistic'}
            The name of the pretrained model for detecting action units. Default is ``JAANET``.
        landmark_model: {'PFLD', 'MobileFaceNet', 'MobileNet'}
            The name of the pretrained model for detecting facial landmarks. Default is ``PFLD``.
        **clargs: dict, optional
            Additional arguments that are passed to the `spectralcluster.SpectralClusterer` class instance.

        Returns
        -------
        A ``FaceExtractor`` class instance that can be used to extract facial features from a video file.

        Notes
        -----
        For details on the available ``au_model`` and ``landmark_model`` arguments, see the documentation of [`pyfeat`](https://py-feat.org/pages/models.html).
        The pretrained action unit models return different outputs: ``JAANET`` returns intensities (0-1) for 12 action units,
        whereas ``svm`` and ``logistic`` return presence/absence (1/0) values for 20 action units.

        """
        self._mtcnn = MTCNN(keep_all=True)
        self._resnet = InceptionResnetV1(
            pretrained='vggface2'
        ).eval()
        self._cluster = SpectralClusterer(**clargs)
        self._pyfeat = feat.detector.Detector(
            au_model=au_model,
            landmark_model=landmark_model
        )


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
            Array containing the bounding boxes of the N detected faces as (x1, y1, x2, y2) coordinates with dimensions (N, 4).
        probs: numpy.ndarray
            Probabilities of the detected faces.

        """

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, probs = self._mtcnn.detect(img, landmarks=False) # pylint: disable=unbalanced-tuple-unpacking

        faces = self._mtcnn.extract(frame, boxes, save_path=None)

        return faces, boxes, probs


    def encode(self, faces):
        """Compute embeddings for face images.

        Parameters
        ----------
        faces: torch.tensor
            Tensor containing N face images with dimensions (N, 3, H, W). H and W must at least be 80 for the encoding to work.

        Returns
        -------
        numpy.ndarray
            Array containing embeddings of the N face images with dimensions (N, 512).

        """
        embeddings = self._resnet(faces).numpy()

        return embeddings


    def identify(self, embeddings):
        """Cluster faces based on their embeddings.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Array containing embeddings of the N face images with dimensions (N, E) where E is the length of the embedding vector.

        Returns
        -------
        numpy.ndarray
            Cluster indices for the N face embeddings.

        """
        labels = np.full((embeddings.shape[0]), np.nan)
        label_finite = np.all(np.isfinite(embeddings), 1)
        labels[label_finite] = self._cluster.predict(embeddings[label_finite, :])

        return labels


    def extract(self, frame, boxes):
        """Detect facial action units and landmarks.

        Parameters
        ----------
        frame: numpy.ndarray
            Array containing the RGB values of a video frame with dimensions (H, W, 3).
        boxes: numpy.ndarray
            Array containing the bounding boxes of the N detected faces as (x1, y1, x2, y2) coordinates with dimensions (N, 4).

        Returns
        -------
        landmarks: numpy.ndarray
            Array containg facial landmarks for N detected faces as (x, y) coordinates with dimensions (N, 68, 2).
        aus: numpy.ndarray
            Array containing action units for N detected faces with dimensions (N, U).
            The number of detected actions units U varies across ``au_model`` specifications.

        """
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0) # convert to 4d

        boxes_list = boxes.reshape(1, -1, 4).tolist()
        landmarks = self._pyfeat.detect_landmarks(frame, boxes_list)
        if self._pyfeat['au_model'].lower() in ['svm', 'logistic']:
            hog, new_landmarks = self._pyfeat._batch_hog( # pylint: disable=protected-access
                frames=frame, detected_faces=boxes_list, landmarks=landmarks
            )
            aus = self._pyfeat.detect_aus(hog, new_landmarks)
        else:
            aus = self._pyfeat.detect_aus(frame, landmarks)

        landmarks_np = np.array(landmarks).squeeze()

        return landmarks_np, aus


    def apply(self, filepath, skip_frames=1, process_subclip=(0, None), show_progress=True): # pylint: disable=too-many-locals
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
            A dictionary with keys-value pairs:
            - `frame`: List of `int` frame indices.
            - `time`: List of `float` timestamps.
            - `face_box`: List of `numpy.ndarray` bounding boxes of detected faces.
            - `face_prob`: List of probabilities of detected faces.
            - `face_landmarks`: List of `numpy.ndarray` facial landmarks.
            - `face_aus`: List of `numpy.ndarray` facial actions units.
            - `face_id`: List of `int` cluster labels of detected faces.

        """
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

            embeddings = [] # Embeddings are separate because they won't be returned

            for i, (t, frame) in tqdm(enumerate(subclip.iter_frames(with_times=True)), disable=not show_progress):
                if i % skip_frames == 0:

                    faces, boxes, probs = self.detect(frame)

                    if faces is None:
                        features['frame'].append(i)
                        features['time'].append(t)
                        features['face_box'].append(np.nan)
                        features['face_prob'].append(np.nan)
                        features['face_landmarks'].append(np.nan)
                        features['face_aus'].append(np.nan)

                        embeddings.append(np.full((self._resnet.last_bn.num_features), np.nan))
                    else:
                        embs = self.encode(faces) # Embeddings per frame
                        landmarks, aus = self.extract(frame, boxes)

                        for box, prob, emb, landmark, au in zip(boxes, probs, embs, landmarks, aus):
                            features['frame'].append(i)
                            features['time'].append(t)
                            features['face_box'].append(box)
                            features['face_prob'].append(prob)
                            features['face_landmarks'].append(landmark)
                            features['face_aus'].append(au)

                            embeddings.append(emb)

            features['face_id'] = self.identify(np.array(embeddings).squeeze()).tolist()

            return features
