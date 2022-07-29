""" Speaker identification classes and methods """

from pyannote.audio import Pipeline


class SpeakerIdentifier:

    def __init__(self) -> 'SpeakerIdentifier':
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")


    def apply(self, filepath, num_speakers=None, verbose=False):
        annotation = self._pipeline(filepath, num_speakers=num_speakers)

        return annotation
