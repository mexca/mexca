""" Test voice feature extraction classes and methods """

import os
import subprocess
import pytest
import numpy as np
from mexca.audio.extraction import FeaturePitchF0, VoiceExtractor, VoiceFeaturesConfig
from mexca.data import VoiceFeatures


@pytest.fixture
def config():
    return VoiceFeaturesConfig()


class TestVoiceFeaturesConfig:
    filename = "test.yaml"
    

    def test_write_read(self, config):
        config.write_yaml(self.filename)
        
        assert os.path.exists(self.filename)

        new_config = config.from_yaml(self.filename)

        assert isinstance(new_config, VoiceFeaturesConfig)

        os.remove(self.filename)


class TestVoiceExtractor:
    time_step = 0.04
    filepath = os.path.join('tests', 'test_files', 'test_video_audio_5_seconds.wav')

    @pytest.fixture
    def voice_extractor(self):
        return VoiceExtractor()


    def test_feature_dict(self):
        with pytest.raises(TypeError):
            _ = VoiceExtractor(features={1: FeaturePitchF0()})
            _ = VoiceExtractor(features={"pitch_fo_hz": [0.0]})


    def test_apply(self, voice_extractor):
        features = voice_extractor.apply(self.filepath, self.time_step)
        assert isinstance(features, VoiceFeatures)
        assert np.issubdtype(np.array(features.frame).dtype, np.int_)
        assert np.all(np.array(features.frame) >= 0) and np.all(np.array(features.frame) <= 125)
        assert len(features.frame) == len(features.pitch_f0_hz)


    def test_cli(self, config):
        config_filepath = "test_config.yaml"
        out_filename = os.path.splitext(os.path.basename(self.filepath))[0] + '_voice_features.json'
        config.write_yaml(config_filepath)
        subprocess.run(['extract-voice', '-f', self.filepath,
                        '-o', '.', '-t', '0.04', "--config-filepath", config_filepath], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
        os.remove(config_filepath)