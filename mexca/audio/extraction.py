"""Extract voice features from an audio file.
"""

import argparse
import logging
import os
import numpy as np
from parselmouth import Sound
from mexca.data import VoiceFeatures
from mexca.utils import ClassInitMessage


class VoiceExtractor:
    """Extract voice features from an audio file.

    Currently, only the voice pitch as the fundamental frequency F0 can be extracted.
    The F0 is calculated using an autocorrelation function with a lower boundary of 75 Hz and an
    upper boudnary of 600 Hz. See the praat 
    `manual <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch___.html>`_ for details.

    """


    def __init__(self):
        self.logger = logging.getLogger('mexca.audio.extraction.VoiceExtractor')
        self.logger.debug(ClassInitMessage())


    def apply(self, filepath: str, time_step: float) -> VoiceFeatures:
        """Extract voice features from an audio file.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        time_step: float
            The interval between time points at which features are extracted.

        Returns
        -------
        VoiceFeatures
            A data class object containing the extracted voice features.

        """
        self.logger.debug('Loading audio file')
        snd = Sound(filepath)
        time = np.arange(snd.start_time, snd.end_time, time_step, dtype=np.float32)
        frame = np.array(time * int(1/time_step), dtype=np.int32)
        
        self.logger.debug('Computing SHS voice pitch')
        pitch = snd.to_pitch()

        pitch_array = np.vectorize(pitch.get_value_at_time)(time)

        return VoiceFeatures(frame=frame.tolist(), time=time.tolist(), pitch_f0=pitch_array.tolist())


def cli():
    """Command line interface for extracting voice features.
    See `extract-voice -h` for details.
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
    