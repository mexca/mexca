""" Test voice feature extraction classes and methods """

import os
import subprocess
import pytest
import numpy as np
from mexca.audio.extraction import FeaturePitchF0, VoiceExtractor
from mexca.data import VoiceFeatures


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


    def test_cli(self):
        out_filename = os.path.splitext(os.path.basename(self.filepath))[0] + '_voice_features.json'
        subprocess.run(['extract-voice', '-f', self.filepath,
                        '-o', '.', '-t', '0.04'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)