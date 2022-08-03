"""Voice features that can be extracted from an audio file using the ``VoiceExtractor`` class.
"""

import numpy as np


class FeaturePitchF0:
    """Calculate the voice pitch as the fundamental frequency F0 from an audio signal.
    """
    def __init__(
        self,
        time_step=None,
        pitch_floor=75.0,
        pitch_ceiling=600.0
    ) -> 'FeaturePitchF0':
        """Create a class instance to calculate the voice pitch.

        Parameters
        ----------
        time_step: float or None, default=None
            The interval between time points at which the voice pitch is calculated.
        pitch_floor: Lower bound frequency of the pitch calculation.
        pitch_ceiling: Upper bound frequency of the pitch calculation.

        Returns
        -------
        A ``FeaturePitchF0`` class instance.

        Notes
        -----
        See the `praat` manual for further details on voice pitch analysis:
        https://www.fon.hum.uva.nl/praat/manual/Intro_4_2__Configuring_the_pitch_contour.html

        """
        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling


    def extract(self, snd, time):
        """Calculate the voice pitch of an audio signal.

        Parameters
        ----------
        snd: parselmouth.Sound
            An object containing the audio signal.
        time: numpy.ndarray
            An array with time points at which the voice pitch is returned.

        Returns
        -------
        numpy.ndarray
            An array with the voice pitch at the times in `time`.

        See Also
        --------
        mexca.audio.extraction.VoiceExtractor: A class to extract voice features from an audio file.

        """
        voice_pitch = snd.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling
        )

        feature = np.vectorize(voice_pitch.get_value_at_time)(time)

        return feature
