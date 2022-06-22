""" Output classes and methods """

from dataclasses import dataclass

@dataclass
class Multimodal:

    def __init__(self, video, audio, text) -> 'Multimodal':
        self.video = video
        self.audio = audio
        self.text = text
