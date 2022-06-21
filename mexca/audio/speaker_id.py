""" Speaker identification classes and methods """

import os
from pyannote.audio import Pipeline

class SpeakerIdentifier:

    def __init__(self) -> 'SpeakerIdentifier':
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")


    def apply(self, filepath, num_speakers=None):
        #check if file exists
        if not os.path.exists(filepath):
            raise Exception("File not found")

        annotation = self._pipeline(filepath,num_speakers=num_speakers)

        return annotation



