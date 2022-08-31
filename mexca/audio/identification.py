"""Identify speech segments and speakers in an audio file.
"""

from pyannote.audio import Pipeline


class SpeakerIdentifier:
    """Extract speech segments and cluster speakers using speaker diarization.
    """
    def __init__(self, num_speakers=None) -> 'SpeakerIdentifier':
        """Create a class instance to apply speaker diarization.

        Parameters
        ----------
        num_speakers: int or None, default=None
            The number of speakers to which speech segments will be assigned during the clustering
            (oracle speakers). If `None`, the number of speakers is estimated from the audio signal.

        Returns
        -------
        A ``SpeakerIdentifier`` class instance.

        """
        self.num_speakers=num_speakers
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")


    def apply(self, filepath):
        """Extract speech segments and speakers.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.

        Returns
        -------
        pyannote.core.Annotation
            A pyannote annotation object that contains detected speech segments and speakers.
            See https://pyannote.github.io/pyannote-core/reference.html#annotation for details.

        """
        annotation = self._pipeline(filepath, num_speakers=self.num_speakers)

        return annotation
