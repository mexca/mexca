"""Identify speech segments and speakers in an audio file.
"""

from pyannote.audio import Pipeline


class SpeakerIdentifier:
    """Extract speech segments and cluster speakers using speaker diarization.

    Parameters
    ----------
    num_speakers: int or None, default=None
        The number of speakers to which speech segments will be assigned during the clustering
        (oracle speakers). If `None`, the number of speakers is estimated from the audio signal.

    Attributes
    ----------
    pyannote_audio

    """
    def __init__(self, num_speakers=None) -> 'SpeakerIdentifier':
        self.num_speakers = num_speakers
        self.pyannote_audio = Pipeline.from_pretrained("pyannote/speaker-diarization")


    @property
    def num_speakers(self):
        return self._num_speakers


    @num_speakers.setter
    def num_speakers(self, new_num_speakers):
        if new_num_speakers:
            if isinstance(new_num_speakers, (int, float)):
                if new_num_speakers >= 2.0:
                    self._num_speakers = int(new_num_speakers)
                else:
                    raise ValueError('Argument "num_speakers" must be >= 2 for speaker identification')
            else:
                raise TypeError('Can only set "num_speakers" to float or int')
        else:
            self._num_speakers = new_num_speakers


    @property
    def pyannote_audio(self):
        """The pyannote speaker diarization pipeline. Must be instance of `Pipeline` class.
        See `pyanote.audio <https://github.com/pyannote/pyannote-audio>`_ for details.
        """
        return self._pyannote_audio


    @pyannote_audio.setter
    def pyannote_audio(self, new_pyannote_audio):
        if isinstance(new_pyannote_audio, Pipeline):
            self._pyannote_audio = new_pyannote_audio
        else:
            raise TypeError('Can only set "pyannote_audio" to instance of "Pipeline" class')
    def __init__(self, use_auth_token=True) -> 'SpeakerIdentifier':
        """Create a class instance to apply speaker diarization.

        Parameters
        ----------
        use_auth_token: bool or str, default=True
            Whether to use the HuggingFace authentication token stored on the machine (if bool) or
            a HuggingFace authentication token with access to the models ``pyannote/speaker-diarization``
            and ``pyannote/segmentation`` (if str).

        Returns
        -------
        A ``SpeakerIdentifier`` class instance.

        Notes
        -----
        This class requires pretrained models for speaker diarization and segmentation from HuggingFace.
        To download the models accept the user conditions on `<hf.co/pyannote/speaker-diarization>`_ and
        `<hf.co/pyannote/segmentation>`_. Then generate an authentication token on `<hf.co/settings/tokens>`_.

        """
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=use_auth_token)


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
        annotation = self.pyannote_audio(filepath, num_speakers=self.num_speakers)

        return annotation
