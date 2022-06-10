""" Face identification classes and methods """

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from mexca.video.annotation import Face
from moviepy.editor import VideoFileClip
from PIL import Image
from spectralcluster import SpectralClusterer
from torchvision import transforms

class FaceIdentifier:
    device = 'cpu'

    def __init__(self) -> 'FaceIdentifier':
        self._transform = transforms.Compose([transforms.ToTensor()])
        self._mtcnn = MTCNN(device=self.device, keep_all=True)
        self._resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()


    def apply(self, filepath):
        with VideoFileClip(filepath, audio=False) as clip:
            faces = []
            frame_idx = 0

            for t, frame in clip.iter_frames(with_times=True):
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                crops, probs = self._mtcnn(img, return_prob=True)

                if crops is not None:
                    embeddings = self._resnet(crops).detach().cpu()
                    for crop, prob, emb in zip(crops, probs, embeddings):
                        face = Face(
                            frame=frame_idx,
                            time=t,
                            array=crop,
                            prob=prob,
                            embeddings=emb,
                            label=None
                        )
                        faces.append(face)

                frame_idx += 1

        return faces


class FaceClassifier(SpectralClusterer):
    def apply(self, faces):
        embeddings = np.stack([face.embeddings for face in faces])
        labels = self.predict(embeddings)
        for face, label in zip(faces, labels):
            face.label = label

        return faces
