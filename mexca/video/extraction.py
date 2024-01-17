"""Facial feature extraction from videos.
"""

import argparse
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances
from spectralcluster import SpectralClusterer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps
from tqdm import tqdm

from mexca.data import EMPTY_VALUE, VideoAnnotation
from mexca.utils import ClassInitMessage, optional_float, optional_int, str2bool
from mexca.video.mefarg import MEFARG

# Package versions are pinned so we can ignore future and deprecation warnings
# Ignore warning from facenet_pytorch
#
# VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
# (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes)
# is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
#    probs = np.array(probs)
warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

# Ignore warning from spectralclusterer
#
# FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4.
# Set the value of `n_init` explicitly to suppress the warning
warnings.simplefilter("ignore", category=FutureWarning)


class NotEnoughFacesError(Exception):
    """Less detected faces than `num_faces`.

    Cannot perform clustering if samples are less than the number of clusters.

    Parameters
    ----------
    msg : str
        Error message.

    """

    def __init__(self, msg: str):
        super().__init__(msg)


# Adapted from pyfeat.data.VideoDataset
# See: https://github.com/cosanlab/py-feat/blob/03337f44e98a915a8488388bce94154d2d5ba73c/feat/data.py#L2154
class VideoDataset(Dataset):
    """Custom torch dataset for a video file.

    Only reads the frame timestamps of the video but not the frames themselves when initialized.
    Decodes the video frame-by-frame.

    Arguments
    ---------
    video_file : str
        Path to the video file.
    skip_frames : int, default=1
        Only load every nth frame.
    start : float, default=0
        Start of the subclip of the video to be loaded (in seconds).
    end : float, optional, default=None
        End of the subclip of the video to be loaded (in seconds).

    Attributes
    ----------
    file_name : str
        Name of the video file.
    video_pts : torch.Tensor
        Timestamps of video frames.
    video_frames_idx : torch.Tensor
        Indices of video frames.
    video_fps: int
        Frames per second.
    video_frames : numpy.ndarray
        Indices of loaded frames.

    """

    def __init__(
        self,
        video_file: str,
        skip_frames: int = 1,
        start: float = 0,
        end: Optional[float] = None,
    ):
        self.logger = logging.getLogger("mexca.video.VideoDataset")
        self.logger.debug("Reading video timestamps")
        video_pts, video_fps = read_video_timestamps(video_file)

        self.file_name = os.path.basename(video_file)
        self._filepath = video_file  # Store path to video for reading frames in __getitem__()

        self.video_fps = video_fps

        # Frame indices of start and end
        start_idx = int(start * video_fps)
        if end is None:
            end_idx = len(video_pts)
        else:
            end_idx = int(end * video_fps)

        self.video_frames_idx = np.arange(start_idx, end_idx, skip_frames)

        # Store timestamps of frames to be loaded
        self.video_pts = torch.Tensor(video_pts)[self.video_frames_idx]

        self.logger.debug(ClassInitMessage())

    @property
    def duration(self) -> float:
        """Duration of the video (read-only)."""
        return len(self) / self.video_fps

    def __len__(self) -> int:
        """Number of video frames."""
        return self.video_pts.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an item from the data set.

        Loads the video frame into memory.

        Parameters
        ----------
        idx : int
            Index of the item in the dataset.

        Returns
        -------
        dict
            Dictionary with 'Image' containing the video frame (T, H, W, C)
            and 'Frame' containing the frame index.

        """
        self.logger.debug("Getting item %s", idx)
        frame_pts = self.video_pts[idx]

        # Read video frame at timestamp of idx
        with warnings.catch_warnings():
            # We can safely ignore this warning because we don't need accurate AV sync:
            # https://github.com/pytorch/vision/issues/1931
            #
            # UserWarning: The pts_unit 'pts' gives wrong results. Please use
            # pts_unit 'sec'
            warnings.simplefilter("ignore", UserWarning)

            image, _, _ = read_video(
                self._filepath,
                start_pts=int(frame_pts.min()),
                end_pts=int(frame_pts.max()),
            )

        # Adjust slice if idx is slice
        if isinstance(idx, slice):
            zero_idx = slice(0, idx.stop - idx.start, idx.step)
            return {
                "Image": image[zero_idx, :, :, :],
                "Frame": self.video_frames_idx[idx],
            }

        return {
            "Image": image.squeeze(),  # Remove first dim
            "Frame": self.video_frames_idx[idx],
        }


class FaceExtractor:
    """Combine steps to extract features from faces in a video file.

    Parameters
    ----------
    num_faces : int, optional
        Number of faces to identify.
    min_face_size : int, default=20
        Minimum face size (in pixels) for face detection in MTCNN.
    thresholds : tuple, default=(0.6, 0.7, 0.7)
        Thesholds for face detection in MTCNN.
    factor : float, default=0.709
        Factor used to create a scaling pyramid of face sizes in MTCNN.
    post_process : bool, default=True
        Whether detected faces are post processed before computing embeddings.
        The post processing standardizes the detected faces.
    select_largest : bool, default=True
        Whether to return the largest face or the one with the highest probability
        if multiple faces are detected.
    selection_method : {None, 'probability', 'largest', 'largest_over_threshold', 'center_weighted_size'}, optional, default=None
        The heuristic used for selecting detected faces. If not `None`, overrides `select_largest`.
    keep_all: bool, default=True
        Whether all faces should be returned in the order of `select_largest`.
    device: torch.device, optional, default=torch.device("cpu")
        The device on which face detection and embedding computations are performed.
    max_cluster_frames : int, optional, default=None
        Maximum number of frames that are used for spectral clustering. If the number of frames exceeds the maximum,
        hierarchical clustering is applied first to reduce the frames to this number. This can reduce the computational
        costs for long videos.
    embeddings_model : {'vggface2', 'casia-webface'}, default='vggface2'
        Pretrained Inception Resnet V1 model for computing face embeddings.
    post_min_face_size : tuple, default=(45.0, 45.0)
        Minimal width and height (in pixels) for filtering out faces after detection.
        This can be useful to exclude small faces before clustering their embeddings
        and can improve clustering performance.


    """

    def __init__(
        self,
        num_faces: Optional[int],
        min_face_size: int = 20,
        thresholds: Tuple[float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        selection_method: Optional[str] = None,
        keep_all: bool = True,
        device: torch.device = torch.device(type="cpu"),
        max_cluster_frames: Optional[int] = None,
        embeddings_model: str = "vggface2",
        post_min_face_size: Tuple[float, float] = (45.0, 45.0),
    ):
        self.logger = logging.getLogger("mexca.video.FaceExtractor")
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.selection_method = selection_method
        self.keep_all = keep_all
        self.logger.debug("Initializing FaceExtractor on device %s", device)
        self.device = device
        self.embeddings_model = embeddings_model
        self.num_faces = num_faces
        self.max_cluster_frames = max_cluster_frames
        self.post_min_face_size = post_min_face_size

        # Lazy initialization: See getter functions
        self._detector = None
        self._encoder = None
        self._clusterer = None
        self._extractor = None

        self.logger.debug(ClassInitMessage())

    # Initialize pretrained models only when needed
    @property
    def detector(self) -> MTCNN:
        """The MTCNN model for face detection and extraction.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch/blob/555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b/models/mtcnn.py#L157>`_ for details.
        """
        if not self._detector:
            self._detector = MTCNN(
                image_size=224,
                min_face_size=self.min_face_size,
                thresholds=self.thresholds,
                factor=self.factor,
                post_process=self.post_process,
                select_largest=self.select_largest,
                selection_method=self.selection_method,
                keep_all=self.keep_all,
                device=self.device,
            )
            self.logger.debug("Initialized MTCNN face detector")
        return self._detector

    # Delete pretrained models when not needed anymore
    @detector.deleter
    def detector(self):
        self._detector = None
        self.logger.debug("Removed MTCNN face detector")

    @property
    def encoder(self) -> InceptionResnetV1:
        """The ResnetV1 model for computing face embeddings.
        See `facenet-pytorch <https://github.com/timesler/facenet-pytorch/blob/555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b/models/inception_resnet_v1.py#L184>`__ for details.
        """
        if not self._encoder:
            self._encoder = InceptionResnetV1(
                pretrained=self.embeddings_model, device=self.device
            ).eval()
            self.logger.debug("Initialized InceptionResnetV1 face encoder")
        return self._encoder

    @encoder.deleter
    def encoder(self):
        self._encoder = None
        self.logger.debug("Removed InceptionResnetV1 face encoder")

    @property
    def clusterer(self) -> SpectralClusterer:
        """The spectral clustering model for identifying faces based on embeddings.
        See `spectralcluster <https://wq2012.github.io/SpectralCluster/>`_ for details.
        """
        if not self._clusterer:
            self._clusterer = SpectralClusterer(
                min_clusters=self.num_faces,
                max_clusters=self.num_faces,
                max_spectral_size=self.max_cluster_frames,
            )
            self.logger.debug("Initialized spectral clusterer")
        return self._clusterer

    @clusterer.deleter
    def clusterer(self):
        self._clusterer = None
        self.logger.debug("Removed spectral clusterer")

    @property
    def extractor(self) -> MEFARG:
        """The MEFARG model for extracting action unit activations.
        See ME-GraphAU `model <https://github.com/CVI-SZU/ME-GraphAU>`_ and
        `paper <https://arxiv.org/abs/2205.01782>`_ for details.
        """
        if not self._extractor:
            self._extractor = MEFARG.from_pretrained(device=self.device)
            self._extractor.eval()
            self.logger.debug(
                "Initialized MEFARG action unit feature extractor"
            )
        return self._extractor

    @extractor.deleter
    def extractor(self):
        self._extractor = None
        self.logger.debug("Removed MEFARG action unit feature extractor")

    def __call__(self, **callargs) -> VideoAnnotation:
        """Alias for `apply`."""
        return self.apply(**callargs)

    @staticmethod
    def _prepare_frame(frame: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extend 3 dim video frames to 4 dim."""
        if frame.ndim == 3:
            if isinstance(frame, np.ndarray):
                return np.expand_dims(frame, 0)

            if isinstance(frame, torch.Tensor):
                return frame.unsqueeze(0)

        if frame.ndim == 4:
            return frame

        raise ValueError(
            f"Argument 'frame' must have dim 3 or 4 but has dim {frame.ndim}"
        )

    def detect(
        self, frame: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        Union[List[np.ndarray], np.ndarray],
        Union[List[np.ndarray], np.ndarray],
        Union[List[np.ndarray], np.ndarray],
    ]:
        """Detect faces in a video frame.

        Parameters
        ----------
        frame: numpy.ndarray or torch.Tensor
            Batch of B frames containing RGB values with dimensions (B, W, H, 3).

        Returns
        -------
        faces: list
            Batch of B tensors containing the N cropped face images from each batched frame with dimensions (N, 3, 160, 160).
            Is `None` if a frame contains no faces.
        boxes: numpy.ndarray or list
            Batch of B bounding boxes of the N detected faces as (x1, y1, x2, y2) coordinates with
            dimensions (B, N, 4). Returns a list if different numbers of faces are detected across batched frames.
            Is `None` if a frame contains no faces.
        probs: numpy.ndarray or list
            Probabilities of the detected faces (B, N).
            Returns a list if different numbers of faces are detected across batched frames.
            Is `None` if a frame contains no faces.
        landmarks: numpy.ndarray or list
            Batch of B facial landmarks for N detected faces as (x, y) coordinates with dimensions (5, 2).
            Is `None` if a frame contains no faces.

        """
        try:
            frame = self._prepare_frame(frame)
        except ValueError as exc:
            self.logger.exception("ValueError: %s", exc)
            raise exc

        self.logger.debug("Detecting faces and facial landmarks")
        boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)

        self.logger.debug("Extracting faces")
        faces = self.detector.extract(frame, boxes, save_path=None)

        return faces, boxes, probs, landmarks

    def encode(self, faces: torch.Tensor) -> np.ndarray:
        """Compute embeddings for face images.

        Parameters
        ----------
        faces: torch.Tensor
            Cropped N face images from a video frame with dimensions (N, 3, H, W). H and W must at least be 80 for
            the encoding to work.

        Returns
        -------
        numpy.ndarray
            Embeddings of the N face images with dimensions (N, 512).

        """

        self.logger.debug("Encoding faces")
        embeddings = self.encoder(faces.to(self.device)).detach().cpu().numpy()

        return embeddings

    def identify(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster faces based on their embeddings.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings of the N face images with dimensions (N, E) where E is the length
            of the embedding vector.

        Returns
        -------
        numpy.ndarray
            Cluster indices for the N face embeddings.

        """
        labels = np.full((embeddings.shape[0]), np.nan)
        labels_finite = np.all(np.isfinite(embeddings), 1)
        embeddings_finite = embeddings[labels_finite, :]

        self.logger.info("Clustering face embeddings")

        try:
            if embeddings_finite.shape[0] < self.num_faces:
                raise NotEnoughFacesError(
                    "Not enough faces detected to perform clustering; consider reducing 'num_faces', 'min_face_size', or 'thresholds'"
                )

        except NotEnoughFacesError as exc:
            self.logger.exception("NotEnoughFacesError: %s", exc)
            raise exc

        labels[labels_finite] = self.clusterer.predict(embeddings_finite)

        return labels

    def extract(
        self,
        frame: Union[np.ndarray, torch.Tensor],
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Detect facial action units activations.

        Parameters
        ----------
        frame: numpy.ndarray or torch.Tensor
            Batch of B frames containing RGB values with dimensions (B, H, W, 3).

        Returns
        -------
        aus: numpy.ndarray or list
            Batch of B action unit activations for N detected faces with dimensions (N, 41).
            Is `None` if a frame contains no faces.

        """
        self.logger.debug("Extracting facial action unit activations")

        aus_list = []

        for frm in frame:
            if frm is None:
                aus_list.append(None)
                continue

            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            transform = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                    normalize,
                ]
            )

            frm = transform(frm)

            with torch.no_grad():
                aus = self.extractor(frm)

            aus_list.append(aus.numpy())

        return np.array(aus_list)

    @staticmethod
    def _compute_centroids(
        embs: np.ndarray, labels: np.ndarray
    ) -> Tuple[List[np.ndarray], Dict[Union[str, int, float], int]]:
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

    def compute_avg_embeddings(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> dict:
        """Computes average embedding vector for each face detected in the video.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Face embeddings.
        labels: numpy.ndarray
            Face labels.

        Returns
        -------
        average embedding dictionary: dict
            Dictionary with keys representing face labels and values representing
            the average embedding vector for each face label.
        """
        self.logger.debug("Computing average embeddings")

        # collect face embeddings and map them to face labels
        face_embedding_dict = {}
        for embedding, label in zip(embeddings, labels):
            # do not take into account nan values for average embedding calculation
            if str(label) != "nan":
                if label in face_embedding_dict:
                    face_embedding_dict[label].append(embedding)
                else:
                    face_embedding_dict[label] = [embedding]

        # calculate average embeddings for each face label
        face_avg_embedding_dict = {}
        for label, embeddings_lst in face_embedding_dict.items():
            face_avg_embedding_dict[label] = np.mean(
                np.array(embeddings_lst), axis=0
            ).tolist()

        return face_avg_embedding_dict

    def compute_confidence(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute face label classification confidence.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Face embeddings.
        labels: numpy.ndarray
            Face labels.

        Returns
        -------
        confidence: numpy.ndarray
            Confidence scores between 0 and 1. Returns `numpy.nan` if no label was assigned to a face.

        """
        self.logger.debug("Computing face clustering confidence")
        centroids, cluster_label_mapping = self._compute_centroids(
            embeddings, labels
        )

        # create empty array with same lenght as labels
        confidence = np.empty_like(labels)

        for i, (frame_embedding, label) in enumerate(zip(embeddings, labels)):
            # if frame is unclassified, assigns zero confidence
            if np.isnan(label):
                confidence[i] = EMPTY_VALUE
            else:
                # compute distance between frame embedding and each centroid
                # frame embedding is put in list becouse it is required by cosine distances API
                # cosine distances returns a list of list, so [0] is taken to get correct shape
                distances = cosine_distances([frame_embedding], centroids)[0]
                # if len(distance) is <= 1, it means that there is only 1 face
                # Clustering confidence is not useful in that case
                if len(distances) <= 1:
                    c = EMPTY_VALUE
                else:
                    # recover index of centroid of cluster
                    cluster_centroid_idx = cluster_label_mapping[label]

                    # distance to the centroid of the cluster to which the frame belongs
                    d1 = distances[cluster_centroid_idx]
                    # mimimum of all other distance

                    d2 = np.min(
                        distances[
                            np.arange(len(distances)) != cluster_centroid_idx
                        ]
                    )

                    # confidence score: 0 if d1 = d2, 1 if d1 = 0
                    c = (d2 - d1) / d2

                    # handle edge cases: d1<d2 by definition, but if the clustering that produced
                    # the labels used a different distance definition, this could not be the case.
                    # These edge cases are low in confidence, i.e., c = 0.
                    c = max(c, 0)

                confidence[i] = c

        return confidence

    @staticmethod
    def _calc_face_size(bbox: List[float]) -> Tuple[float, float]:
        """Calculate the size of a detected face bounding box."""

        box_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if box_size[0] > 0.0 and box_size[1] > 0.0:
            return box_size

        raise RuntimeError(f"Bounding box had a non positive size {box_size}")

    def _check_face_size(self, box_size: Tuple[float, float]) -> bool:
        return (
            box_size[0] >= self.post_min_face_size[0]
            or box_size[1] >= self.post_min_face_size[1]
        )

    def apply(  # pylint: disable=too-many-locals
        self,
        filepath: str,
        batch_size: int = 1,
        skip_frames: int = 1,
        process_subclip: Tuple[Optional[float]] = (0, None),
        show_progress: bool = True,
    ) -> VideoAnnotation:
        """Apply multiple steps to extract features from faces in a video file.

        This method subsequently calls other methods for each frame of a video file to detect
        and cluster faces. It also extracts facial landmarks and action units.

        Parameters
        ----------
        filepath: str
            Path to the video file.
        batch_size: int, default=1
            Size of the batch of video frames that are loaded and processed at the same time.
        skip_frames: int, default=1
            Only process every nth frame, starting at 0.
        process_subclip: tuple, default=(0, None)
            Process only a part of the video clip. Must be the start and end of the subclip in seconds.
        show_progress: bool, default=True
            Enables the display of a progress bar.

        Returns
        -------
        VideoAnnotation
            A data class object with extracted facial features.

        """
        video_dataset = VideoDataset(
            filepath,
            skip_frames=skip_frames,
            start=process_subclip[0],
            end=process_subclip[1],
        )

        batch_data_loader = DataLoader(video_dataset, batch_size=batch_size)

        # Store features
        annotation = VideoAnnotation(filename=filepath)

        embeddings = (
            []
        )  # Embeddings are separate because they won't be returned

        self.logger.info(
            "Detecting and encoding faces, extracting facial features"
        )
        for b, batch in tqdm(
            enumerate(batch_data_loader),
            total=len(batch_data_loader),
            disable=not show_progress,
        ):
            self.logger.debug("Processing batch %s", b)
            faces, boxes, probs, landmarks = self.detect(batch["Image"])

            aus = self.extract(faces)

            # If no faces were detected in batch
            for i, frame in enumerate(batch["Frame"]):
                self.logger.debug("Processing frame %s", int(frame))
                if boxes[i] is not None:
                    box_invalid = [
                        not self._check_face_size(self._calc_face_size(box))
                        for box in boxes[i]
                    ]

                if faces[i] is None or all(box_invalid):
                    self.logger.debug(
                        "No faces detected in frame %s", int(frame)
                    )
                    embeddings.append(
                        np.full((self.encoder.last_bn.num_features), np.nan)
                    )
                    annotation.frame.append(int(frame))
                    annotation.face_box.append(EMPTY_VALUE)
                    annotation.face_landmarks.append(EMPTY_VALUE)
                    annotation.face_prob.append(EMPTY_VALUE)
                    annotation.face_aus.append(EMPTY_VALUE)

                else:
                    self.logger.debug(
                        "%s faces detected in frame %s",
                        len(faces[i]),
                        int(frame),
                    )

                    embs = self.encode(faces[i])
                    for k, (box, box_v, prob, landmark, au, emb) in enumerate(
                        zip(
                            boxes[i],
                            box_invalid,
                            probs[i],
                            landmarks[i],
                            aus[i],
                            embs,
                        )
                    ):
                        self.logger.debug("Processing face %s", k)

                        # If bbox is smaller than min, skip face
                        if box_v:
                            self.logger.debug(
                                "Skipped face %s because its size %s is smaller than threshold",
                                k,
                                box_v,
                            )
                            continue
                        # Convert everything to lists to make saving and loading easier
                        annotation.frame.append(int(frame))
                        annotation.face_box.append(box.tolist())
                        annotation.face_landmarks.append(landmark.tolist())
                        annotation.face_prob.append(
                            float(prob)
                        )  # convert to build-in float to avoid trouble when saving to json
                        annotation.face_aus.append(au.tolist())

                        embeddings.append(emb)

        # Delete pretrained models to save memory
        del self.detector
        del self.encoder
        del self.extractor

        face_label = self.identify(np.asarray(embeddings).squeeze()).tolist()
        annotation.face_label = [
            lbl if np.isfinite(lbl) else EMPTY_VALUE for lbl in face_label
        ]
        annotation.face_average_embeddings = self.compute_avg_embeddings(
            np.asarray(embeddings), np.asarray(face_label)
        )
        annotation.face_confidence = self.compute_confidence(
            np.asarray(embeddings), np.asarray(face_label)
        ).tolist()
        annotation.time = (
            np.array(annotation.frame) / video_dataset.video_fps
        ).tolist()

        del self.clusterer

        return annotation


def cli():
    """Command line interface for extracting facial features.
    See `extract-faces -h` for details.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-f", "--filepath", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument(
        "--num-faces", type=optional_int, default=None, dest="num_faces"
    )

    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument(
        "--skip-frames", type=int, default=1, dest="skip_frames"
    )
    parser.add_argument(
        "--process-subclip",
        type=optional_float,
        nargs=2,
        default=[0, None],
        dest="process_subclip",
    )
    parser.add_argument(
        "--show-progress", type=str2bool, default=True, dest="show_progress"
    )

    parser.add_argument(
        "--min-face-size", type=int, default=20, dest="min_face_size"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs=3, default=[0.6, 0.7, 0.7]
    )
    parser.add_argument("--factor", type=float, default=0.709)
    parser.add_argument(
        "--post-process", type=str2bool, default=True, dest="post_process"
    )
    parser.add_argument(
        "--select-largest", type=str2bool, default=True, dest="select_largest"
    )
    parser.add_argument(
        "--selection-method", type=str, default=None, dest="selection_method"
    )
    parser.add_argument(
        "--keep-all", type=str2bool, default=True, dest="keep_all"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max-cluster-frames",
        type=optional_int,
        default=None,
        dest="max_cluster_frames",
    )
    parser.add_argument(
        "--embeddings-model",
        type=str,
        default="vggface2",
        dest="embeddings_model",
    )

    args = parser.parse_args().__dict__

    filepath: str = args.pop("filepath")
    outdir: str = args.pop("outdir")
    batch_size: int = args.pop("batch_size")
    skip_frames: int = args.pop("skip_frames")
    process_subclip: list = args.pop("process_subclip")
    show_progress: bool = args.pop("show_progress")

    extractor = FaceExtractor(**args)

    output = extractor.apply(
        filepath=filepath,
        batch_size=batch_size,
        skip_frames=skip_frames,
        process_subclip=process_subclip,
        show_progress=show_progress,
    )

    output.write_json(
        os.path.join(
            outdir,
            os.path.splitext(os.path.basename(filepath))[0]
            + f"_{output.serialization_name()}.json",
        )
    )


if __name__ == "__main__":
    cli()
