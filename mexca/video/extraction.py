""" Facial feature extraction class and methods """

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
    device = 'cpu'

    def __init__(self, au_model='JAANET', landmark_model='PFLD', **clargs) -> 'FaceExtractor':
        self._mtcnn = MTCNN(device=self.device, keep_all=True)
        self._resnet = InceptionResnetV1(
            pretrained='vggface2',
            device=self.device
        ).eval()
        self._cluster = SpectralClusterer(**clargs)
        self._pyfeat = feat.detector.Detector(
            au_model=au_model,
            landmark_model=landmark_model
        )


    def detect(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, probs = self._mtcnn.detect(img, landmarks=False) # pylint: disable=unbalanced-tuple-unpacking

        faces = self._mtcnn.extract(frame, boxes, save_path=None)

        return faces, boxes, probs


    def encode(self, faces):
        embeddings = self._resnet(faces)

        return embeddings


    def identify(self, embeddings):
        labels = self._cluster.predict(embeddings.squeeze())

        return labels


    def extract(self, frame, boxes):
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

        return landmarks, aus


    def apply(self, filepath, skip_frames=1, verbose=False): # pylint: disable=too-many-locals
        """
        Apply py-feat video pipeline

        Parameters
        ---------------------------------
        filepath: str,
            audio file path
        skip_frames: float,
            Tells detector to process every nth frame. Default to 1.
        verbose: bool,
            Enables the display of a progress bar. Default to False.
        """
        with VideoFileClip(filepath, audio=False, verbose=False) as clip:
            features = {
                'frame': [],
                'time': [],
                'face_box': [],
                'face_prob': [],
                'face_landmarks': [],
                'face_aus': []
            }

            embeddings = [] # Embeddings are separate because they won't be returned

            frame_idx = 0
            for i, (t, frame) in tqdm(enumerate(clip.iter_frames(with_times=True)), disable=not verbose):
                if i % skip_frames == 0:

                    faces, boxes, probs = self.detect(frame)

                    if faces is not None:
                        embs = self.encode(faces).numpy() # Embeddings per frame
                        landmarks, aus = self.extract(frame, boxes)
                        landmarks_np = np.array(landmarks).squeeze()

                        for box, prob, emb, landmark, au in zip(boxes, probs, embs, landmarks_np, aus):
                            features['frame'].append(frame_idx)
                            features['time'].append(t)
                            features['face_box'].append(box)
                            features['face_prob'].append(prob)
                            features['face_landmarks'].append(landmark)
                            features['face_aus'].append(au)

                            embeddings.append(emb)
                # processed frames are indexed with original video index
                frame_idx += 1

            features['face_id'] = self.identify(np.array(embeddings)).tolist()

            return features
