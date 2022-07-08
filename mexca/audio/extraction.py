""" Voice feature extraction classes and methods """

from decimal import Decimal
import numpy as np
from parselmouth import Sound
import mexca.audio.features
from mexca.core.exceptions import TimeStepError
from mexca.core.exceptions import TimeStepWarning


class VoiceExtractor:
    def __init__(self, time_step=None, features=None) -> 'VoiceExtractor':
        self.time_step = time_step

        if features:
            self.features = features
        else:
            self.set_default_features()


    def set_default_features(self):
        self.features = {
            'pitchF0': mexca.audio.features.FeaturePitchF0()
        }


    def extract_features(self, filepath, time):
        snd = Sound(filepath)

        if not time and not self.time_step:
            raise TimeStepError()

        if not time:
            end_time = snd.get_end_time()
            not_processed = Decimal(end_time)%Decimal(self.time_step) # Use 'Decimal' to avoid issues with float representation

            if not_processed > 0.0:
                TimeStepWarning(f'Length of file is not a multiple of "time_step": {not_processed}s at the end of the file will not be processed')

            time = np.arange(start=0.0, stop=end_time, step=self.time_step)

        voice_features = {
            'time': time
        }

        for feature in self.features:
            feature_method = self.features[feature]
            voice_features[feature] = feature_method.extract(snd, time)

        return voice_features
