""" Voice feature extraction classes and methods """

from parselmouth import Sound
import mexca.audio.features
from mexca.core.exceptions import TimeStepError
from mexca.core.utils import create_time_var_from_step


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
            time = create_time_var_from_step(self.time_step, end_time)

        voice_features = {
            'time': time
        }

        for feature in self.features:
            feature_method = self.features[feature]
            voice_features[feature] = feature_method.extract(snd, time)

        return voice_features
