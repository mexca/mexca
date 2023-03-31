"""Speech segment and speaker identification.
"""

import argparse
import logging
import os
from typing import Optional, Union
from pyannote.audio import Pipeline
from mexca.data import SpeakerAnnotation
from mexca.utils import ClassInitMessage, bool_or_str, optional_int


class AuthenticationError(Exception):
    """Failed authentication to HuggingFace Hub.

    Parameters
    ----------
    msg : str
        Error message.

    """

    def __init__(self, msg: str):
        super().__init__(msg)


class SpeakerIdentifier:
    """Identify speech segments and cluster speakers using speaker diarization.

    Wrapper class for ``pyannote.audio.SpeakerDiarization``.

    Parameters
    ----------
    num_speakers : int, optional
        Number of speakers to which speech segments will be assigned during the clustering
        (oracle speakers). If `None`, the number of speakers is estimated from the audio signal.
    use_auth_token : bool or str, default=True
        Whether to use the HuggingFace authentication token stored on the machine (if bool) or
        a HuggingFace authentication token with access to the models ``pyannote/speaker-diarization``
        and ``pyannote/segmentation`` (if str).

    Notes
    -----
    This class requires pretrained models for speaker diarization and segmentation from HuggingFace.
    To download the models accept the user conditions on `<hf.co/pyannote/speaker-diarization>`_ and
    `<hf.co/pyannote/segmentation>`_. Then generate an authentication token on `<hf.co/settings/tokens>`_.

    """

    def __init__(
        self,
        num_speakers: Optional[int] = None,
        use_auth_token: Union[bool, str] = True,
    ):
        self.logger = logging.getLogger("mexca.audio.identification.SpeakerIdentifier")
        self.num_speakers = num_speakers
        self.use_auth_token = use_auth_token
        # Lazy initialization
        self._pipeline = None
        self.logger.debug(ClassInitMessage())

    # Initialize pretrained models only when needed
    @property
    def pipeline(self) -> Pipeline:
        """The pretrained speaker diarization pipeline.
        See `pyannote.audio.SpeakerDiarization <https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L56>`_ for details.
        """
        if not self._pipeline:
            try:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization", use_auth_token=self.use_auth_token
                )

            except EnvironmentError as exc:
                self.logger.exception("EnvironmentError: %s", exc)
                raise exc

            try:
                if self._pipeline is None:
                    raise AuthenticationError(
                        'Could not download pretrained "pyannote/speaker-diarization" pipeline; please provide a valid authentication token'
                    )

            except AuthenticationError as exc:
                self.logger.exception("Error: %s", exc)
                raise exc

            self.logger.debug("Initialized speaker diarization pipeline")

        return self._pipeline

    # Delete pretrained models when not needed anymore
    @pipeline.deleter
    def pipeline(self):
        self._pipeline = None
        self.logger.debug("Removed speaker diarization pipeline")

    def apply(self, filepath: str) -> SpeakerAnnotation:
        """Identify speech segments and speakers.

        Parameters
        ----------
        filepath : str
            Path to the audio file.

        Returns
        -------
        SpeakerAnnotation
            A data class object that contains detected speech segments and speakers.

        """

        annotation = self.pipeline(filepath, num_speakers=self.num_speakers)

        del self.pipeline

        self.logger.info("Detected %s speakers", len(annotation.labels()))
        self.logger.debug("Detected speaker chart: %s", annotation.chart())

        return SpeakerAnnotation.from_pyannote(
            annotation.rename_labels(generator="int")
        )


def cli():
    """Command line interface for identifying speech segments and speakers.
    See `identify-speakers -h` for details.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-f", "--filepath", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument(
        "--num-speakers", type=optional_int, default=None, dest="num_speakers"
    )
    parser.add_argument(
        "--use-auth-token", type=bool_or_str, default=True, dest="use_auth_token"
    )

    args = parser.parse_args().__dict__

    identifier = SpeakerIdentifier(
        num_speakers=args["num_speakers"], use_auth_token=args["use_auth_token"]
    )

    output = identifier.apply(args["filepath"])

    output.write_rttm(
        os.path.join(
            args["outdir"],
            os.path.splitext(os.path.basename(args["filepath"]))[0]
            + "_audio_annotation.rttm",
        )
    )


if __name__ == "__main__":
    cli()
