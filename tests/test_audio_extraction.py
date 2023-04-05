""" Test voice feature extraction classes and methods """

import os
import subprocess
import numpy as np
from mexca.audio.extraction import VoiceExtractor, VoiceFeatures


class TestVoiceExtractor:
    time_step = 0.04
    extractor = VoiceExtractor()
    filepath = os.path.join('tests', 'test_files', 'test_video_audio_5_seconds.wav')


    def test_apply(self):
        features = self.extractor.apply(self.filepath, self.time_step)
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