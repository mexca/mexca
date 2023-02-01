""" Test speaker identification classes and methods """

import os
import subprocess
import pytest
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline
from mexca.audio import SpeakerIdentifier
from mexca.audio.identification import AuthenticationError
from mexca.data import SegmentData, SpeakerAnnotation


class TestSpeakerIdentifier:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )


    @staticmethod
    def check_rttm_annotation(annotation):
        assert isinstance(annotation, SpeakerAnnotation)
                
        for seg in annotation.items():
            assert isinstance(seg.data, SegmentData)
            assert isinstance(seg.begin, float)
            assert 5.0 >= seg.begin >= 0.0
            assert isinstance(seg.end, float)
            assert 5.0 >= seg.end >= 0.0
            assert isinstance(seg.data.name, str)


    @pytest.fixture
    def speaker_identifier(self):
        return SpeakerIdentifier(use_auth_token=self.use_auth_token)


    def test_lazy_init(self, speaker_identifier):
        assert not speaker_identifier._pipeline
        assert isinstance(speaker_identifier.pipeline, Pipeline)
        del speaker_identifier.pipeline
        assert not speaker_identifier._pipeline


    hf_folder = HfFolder()


    @pytest.mark.skipif(
        hf_folder.get_token() is not None,
        reason='Running test locally and a token might be cached'
    )
    def test_environment_error(self):
        with pytest.raises(EnvironmentError):
            speaker_identifier = SpeakerIdentifier(use_auth_token=True)
            speaker_identifier.pipeline


    def test_authentication_error(self):
        with pytest.raises(AuthenticationError):
            speaker_identifier = SpeakerIdentifier(use_auth_token='')
            speaker_identifier.pipeline

        with pytest.raises(AuthenticationError):
            speaker_identifier = SpeakerIdentifier(use_auth_token='hf_riskdkdifseflskefssfjsfsjfsfjsfsfj') # dummy token
            speaker_identifier.pipeline


    def test_apply(self, speaker_identifier):
        annotation = speaker_identifier.apply(self.filepath)

        self.check_rttm_annotation(annotation)


    def test_apply_num_speakers(self):
        num_speakers = 2
        speaker_identifier = SpeakerIdentifier(
            num_speakers=num_speakers,
            use_auth_token=self.use_auth_token,
        )
        annotation = speaker_identifier.apply(self.filepath)

        self.check_rttm_annotation(annotation)


    def test_cli(self):
        out_filename = os.path.splitext(os.path.basename(self.filepath))[0] + '_audio_annotation.rttm'
        subprocess.run(['identify-speakers', '-f', self.filepath,
                        '-o', '.', '--num-speakers', '2',
                        '--use-auth-token', self.use_auth_token], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
