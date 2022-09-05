"""Voice features that can be extracted from an audio file using the ``VoiceExtractor`` class.
"""

import numpy as np


class FeaturePitchF0:
    """Calculate the voice pitch as the fundamental frequency F0 from an audio signal.

    Parameters
    ----------
    time_step: float or None, default=None
        The interval between time points at which the voice pitch is calculated.
    pitch_floor: Lower bound frequency of the pitch calculation.
    pitch_ceiling: Upper bound frequency of the pitch calculation.

    Notes
    -----
    See the `praat
    <https://www.fon.hum.uva.nl/praat/manual/Intro_4_2__Configuring_the_pitch_contour.html>`_
    manual for further details on voice pitch analysis.

    """
    def __init__(
        self,
        time_step=None,
        pitch_floor=75.0,
        pitch_ceiling=600.0
    ) -> 'FeaturePitchF0':
        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling


    @property
    def time_step(self):
        return self._time_step


    @time_step.setter
    def time_step(self, new_time_step):
        if new_time_step:
            if isinstance(new_time_step, (float, int)):
                if new_time_step >= 0.0:
                    self._time_step = new_time_step
                else:
                    raise ValueError('Can only set "time_step" to values >= zero')
            else:
                raise TypeError('Can only set "time_step" to float, int, or None')
        else:
            self._time_step = new_time_step


    @property
    def pitch_floor(self):
        return self._pitch_floor


    @pitch_floor.setter
    def pitch_floor(self, new_pitch_floor):
        if isinstance(new_pitch_floor, float):
            if new_pitch_floor >= 0.0:
                self._pitch_floor = new_pitch_floor
            else:
                raise ValueError('Can only set "pitch_floor" to values >= zero')
        else:
            raise TypeError('Can only set "pitch_floor" to float')


    @property
    def pitch_ceiling(self):
        return self._pitch_ceiling


    @pitch_ceiling.setter
    def pitch_ceiling(self, new_pitch_ceiling):
        if isinstance(new_pitch_ceiling, float):
            if new_pitch_ceiling >= 0.0:
                self._pitch_ceiling = new_pitch_ceiling
            else:
                raise ValueError('Can only set "pitch_ceiling" to values >= zero')
        else:
            raise TypeError('Can only set "pitch_ceiling" to float')


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
