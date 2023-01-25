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


    def apply(self, filepath: str, time_step: float, skip_frames: int = 1) -> VoiceFeatures:
        """Extract voice features from an audio file.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        time_step: float
            The interval between time points at which features are extracted.
        skip_frames: int
            Only process every nth frame, starting at 0.

        Returns
        -------
        VoiceFeatures
            A data class object containing the extracted voice features.

        """
        self.logger.debug('Loading audio file')
        self.logger.debug('Extracting features with time step: %s', time_step)
        snd = Sound(filepath)
        self.logger.debug('End time: %s', snd.xmax)
        time = np.arange(snd.xmin, snd.xmax, time_step, dtype=np.float32)
        frame = np.array((time / time_step) * skip_frames, dtype=np.int32)
        
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
    parser.add_argument('--skip-frames', type=int, default=1, dest='skip_frames')

    args = parser.parse_args().__dict__

    extractor = VoiceExtractor()

    output = extractor.apply(args['filepath'], time_step=args['time_step'], skip_frames=args['skip_frames'])

    output.write_json(os.path.join(args['outdir'], os.path.splitext(os.path.basename(args['filepath']))[0] + '_voice_features.json'))


if __name__ == '__main__':
    cli()
    