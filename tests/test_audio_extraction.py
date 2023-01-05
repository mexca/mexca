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
        assert np.all(np.array(features.time) >= 0.0) and np.all(np.array(features.time) <= 5.0)
        assert len(features.time) == len(features.pitch_F0)


    def test_cli(self):
        out_filename = os.path.basename(self.filepath) + '.json'
        subprocess.run(['extract-voice', '-f', self.filepath,
                        '-o', '.', '-t', '0.04'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)