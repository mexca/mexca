""" Face identification classes and methods """

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from moviepy.editor import VideoFileClip
from PIL import Image
from torchvision import transforms

class FaceIdentifier:
    device = 'cpu'

    def __init__(self) -> 'FaceIdentifier':
        self._transform = transforms.Compose([transforms.ToTensor()])
        self._mtcnn = MTCNN(device=self.device)
        self._resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()


    def apply(self, filepath):
        with VideoFileClip(filepath, audio=False) as clip:
            faces = []

            for frame in clip.iter_frames():
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, probs = self._mtcnn.detect(img)

                if boxes is not None:
                    crops = torch.stack([self._transform(img.crop(tuple(box))) for box in boxes])
                    embeddings = self._resnet.forward(crops).detach().cpu()
                    faces.append({
                        'boxes': [box.tolist() for box in boxes],
                        'probs': [prob.tolist() for prob in probs],
                        'embeddings': [emb.tolist() for emb in embeddings]
                    })

        return faces
