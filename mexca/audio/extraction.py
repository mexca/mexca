"""Extract voice features from an audio file.
Currently, only the voice pitch as the fundamental frequency F0 can be extracted.
"""

from parselmouth import Sound
import mexca.audio.features
from mexca.core.exceptions import TimeStepError
from mexca.core.utils import create_time_var_from_step


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
    def __init__(self, time_step=None, features=None) -> 'VoiceExtractor':
        self.time_step = time_step

        if features:
            self.features = features
        else:
            self.set_default_features()


    @property
    def time_step(self):
        """Interval between time points at which voice features are extracted. Must be `float`.
        """
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
    def features(self):
        """Voice features to be extracted. Must be `dict`.
        """
        return self._features


    @features.setter
    def features(self, new_features):
        if isinstance(new_features, dict):
            self._features = new_features
        else:
            raise TypeError('Can only set "features" to dict class instances')


    def set_default_features(self) -> None:
        """Set `feature` attribute to the default feature classes.
        Currently, this only includes `mexca.audio.features.PitchF0`.
        """
        self.features = {
            'pitchF0': mexca.audio.features.FeaturePitchF0()
        }


    def extract_features(self, filepath, time):
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
        if not time and not self.time_step:
            raise TimeStepError()

        snd = Sound(filepath)

        if not time:
            end_time = snd.get_end_time()
            time = create_time_var_from_step(self.time_step, end_time)

        voice_features = {
            'time': time
        }

        for feature in self.features:
            feature_method = self.features[feature]
            voice_features[feature] = feature_method.extract(snd, time)

        return voice_features
