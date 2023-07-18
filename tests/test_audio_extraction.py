""" Test voice feature extraction classes and methods """

import os
import subprocess

import numpy as np
import pytest

from mexca.audio.extraction import (
    BaseFeature,
    FeaturePitchF0,
    VoiceExtractor,
    VoiceFeaturesConfig,
)
from mexca.data import VoiceFeatures


class TestBaseFeature:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            _ = BaseFeature()


class TestVoiceExtractor:
    time_step = 0.04
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )

    @pytest.fixture
    def config(self):
        return VoiceFeaturesConfig()

    @pytest.fixture
    def voice_extractor(self):
        return VoiceExtractor()

    @pytest.fixture
    def voice_extractor_config(self, config):
        return VoiceExtractor(config=config)

    def test_feature_dict(self):
        with pytest.raises(TypeError):
            _ = VoiceExtractor(features={1: FeaturePitchF0()})
            _ = VoiceExtractor(features={"pitch_fo_hz": [0.0]})

    def test_apply(self, voice_extractor):
        features = voice_extractor.apply(self.filepath, self.time_step)
        assert isinstance(features, VoiceFeatures)
        assert np.issubdtype(np.array(features.frame).dtype, np.int_)
        assert np.all(np.array(features.frame) >= 0) and np.all(
            np.array(features.frame) <= 125
        )
        assert len(features.frame) == len(features.pitch_f0_hz)

    def test_apply_config(self, voice_extractor_config):
        features = voice_extractor_config.apply(self.filepath, self.time_step)
        assert isinstance(features, VoiceFeatures)
        assert np.issubdtype(np.array(features.frame).dtype, np.int_)
        assert np.all(np.array(features.frame) >= 0) and np.all(
            np.array(features.frame) <= 125
        )
        assert len(features.frame) == len(features.pitch_f0_hz)

    def test_cli(self, config):
        config_filepath = "test_config.yaml"
        out_filename = (
            os.path.splitext(os.path.basename(self.filepath))[0]
            + "_voice_features.json"
        )
        config.write_yaml(config_filepath)
        subprocess.run(
            [
                "extract-voice",
                "-f",
                self.filepath,
                "-o",
                ".",
                "-t",
                "0.04",
                "--config-filepath",
                config_filepath,
            ],
            check=True,
        )
        assert os.path.exists(out_filename)
        os.remove(out_filename)
        os.remove(config_filepath)
