""" Facial feature extraction class and methods """

import cv2
import feat
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from moviepy.editor import VideoFileClip
from PIL import Image
from spectralcluster import SpectralClusterer

class FaceExtractor:
    device = 'cpu'

    def __init__(self, **kwargs) -> 'FaceExtractor':
        self._mtcnn = MTCNN(device=self.device, keep_all=True)
        self._resnet = InceptionResnetV1(
            pretrained='vggface2',
            device=self.device
        ).eval()
        self._cluster = SpectralClusterer(**kwargs)
        self._pyfeat = feat.detector.Detector(
            au_model='JAANET',
            landmark_model='PFLD'
        )


    def detect(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, probs = self._mtcnn.detect(img, landmarks=False)

        faces = self._mtcnn.extract(frame, boxes, save_path=None)

        return faces, boxes, probs


    def identify(self, faces):
        if faces is not None:
            embeddings = self._resnet(faces).detach().cpu()
            labels = self._cluster.predict(embeddings.numpy())

        else:
            labels = np.array(np.nan)

        return labels


    def extract(self, frame, boxes):
        if boxes is not None:
            boxes_list = boxes.reshape(1, -1, 4).tolist()
            landmarks = self._pyfeat.detect_landmarks(frame, boxes_list)
            aus = self._pyfeat.detect_aus(frame, landmarks)
        else:
            landmarks = np.nan
            aus = np.nan

        return landmarks, aus


    def apply(self, filepath):
        with VideoFileClip(filepath, audio=False) as clip:
            features = {
                'frame': [],
                'time': [],
                'box': [],
                'prob': [],
                'landmarks': [],
                'aus': [],
                'label': []
            }


            frame_idx = 0
            for t, frame in clip.iter_frames(with_times=True):

                faces, boxes, probs = self.detect(frame)
                if faces is not None:
                    labels = [i for i in range(len(faces))] #self.identify(faces)
                    landmarks, aus = self.extract(frame, boxes)
                    landmarks_np = np.array(landmarks).squeeze()
                else:
                    boxes = [np.nan]
                    probs = [np.nan]
                    labels = [np.nan]
                    landmarks_np = [np.nan]
                    aus = [np.nan]


                for box, prob, label, landmark, au in zip(boxes, probs, labels, landmarks_np, aus):
                    features['frame'].append(frame_idx)
                    features['time'].append(t)
                    features['box'].append(box)
                    features['prob'].append(prob)
                    features['landmarks'].append(landmark)
                    features['aus'].append(au)
                    features['label'].append(label)

                frame_idx += 1

            labels = self.identify(faces)

            return features
