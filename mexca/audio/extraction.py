"""Extract voice features from an audio file.
Currently, only the voice pitch as the fundamental frequency F0 can be extracted.
"""

import argparse
import os
import numpy as np
from parselmouth import Sound
from mexca.data import VoiceFeatures


class VoiceExtractor:
    """Extract voice features from an audio file.

    Parameters
    ----------
    time_step: float or None, default=None
        The interval between time points at which features are extracted.
    features: dict or None, default=None
        A dictionary of features with keys as the features names for the output and values as feature class instances.

    Attributes
    ----------
    time_step
    features

    See Also
    --------
    mexca.audio.features.PitchF0 : Extract the voice pitch as the fundamental frequency F0.

    Notes
    -----
    If `features=None` (the default), the class will be initiated to extract default voice features using the `set_default_features` method.
    Currently, only a class for extracting the voice pitch as the fundamental frequency F0 is available as `mexca.audio.features.PitchF0`.
    A tutorial on how to create a custom feature class will follow soon (TODO).

    """


    def apply(self, filepath: str, time_step: float = 0.023):
        """Extract voice features from an audio file.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.
        time: list or numpy.ndarray or None
            A list of floats or array containing time points at which voice features are extracted.

        Returns
        -------
        dict
            A dictionary with keys corresponding to the names of the extracted features
            and values as arrays containing the extracted feature values.

        Raises
        ------
        TimeStepError
            If `time=None` and `VoiceExtractor.time_step=None`.
            Either the time points for extraction must be supplied or
            the time points must be constructed with with the `time_step` attribute.

        """
        snd = Sound(filepath)
        time = np.arange(snd.start_time, snd.end_time, time_step)
        
        pitch = snd.to_pitch_shs()

        pitch_array = np.vectorize(pitch.get_value_at_time)(time)

        return VoiceFeatures(time=time.tolist(), pitch_F0=pitch_array.tolist())


def cli():
    """Command line interface for extracting voice features.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('-t', '--time-step', type=float, dest='time_step')

    args = parser.parse_args().__dict__

    extractor = VoiceExtractor()

    output = extractor.apply(args['filepath'], time_step=args['time_step'])

    output.write_json(os.path.join(args['outdir'], os.path.splitext(os.path.basename(args['filepath']))[0] + '_voice_features.json'))


if __name__ == '__main__':
    cli()
    