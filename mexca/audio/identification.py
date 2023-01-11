"""Speech segment and speaker identification.
"""

import argparse
import os
from typing import Optional, Union
from pyannote.audio import Pipeline
from mexca.data import RttmAnnotation
from mexca.utils import optional_int, str2bool


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

    Attributes
    ----------
    pipeline : pyannote.audio.Pipeline
        The pretrained speaker diarization pipeline.
        See `pyannote.audio.SpeakerDiarization <https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L56>`_ for details.

    Notes
    -----
    This class requires pretrained models for speaker diarization and segmentation from HuggingFace.
    To download the models accept the user conditions on `<hf.co/pyannote/speaker-diarization>`_ and
    `<hf.co/pyannote/segmentation>`_. Then generate an authentication token on `<hf.co/settings/tokens>`_.

    """
    def __init__(self,
        num_speakers: Optional[int] = None,
        use_auth_token: Union[bool, str] = True
    ):
        self.num_speakers = num_speakers
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=use_auth_token
        )


    def apply(self, filepath: str) -> RttmAnnotation:
        """Identify speech segments and speakers.

        Parameters
        ----------
        filepath : str
            Path to the audio file.

        Returns
        -------
        pyannote.core.Annotation
            A pyannote annotation object that contains detected speech segments and speakers.

        """

        annotation = self.pipeline(filepath, num_speakers=self.num_speakers)

        return RttmAnnotation.from_pyannote(
            annotation.rename_labels(generator='int').rename_tracks(generator='int')
        )


def cli():
    """Command line interface for identifying speech segments and speakers.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('--num-speakers', type=optional_int, default=None, dest='num_speakers')
    parser.add_argument('--use-auth-token', type=str, default=True, dest='use_auth_token')

    args = parser.parse_args().__dict__

    identifier = SpeakerIdentifier(
        num_speakers=args['num_speakers'],
        use_auth_token=args['use_auth_token']
    )

    output = identifier.apply(args['filepath'])

    output.write_rttm(os.path.join(
        args['outdir'],
        os.path.splitext(os.path.basename(args['filepath']))[0] + 'audio_annotation.rttm'
    ))


if __name__ == '__main__':
    cli()
    