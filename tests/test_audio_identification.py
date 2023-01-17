""" Test speaker identification classes and methods """

import os
import subprocess
import pytest
from pyannote.audio import Pipeline
from mexca.audio import SpeakerIdentifier
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
                        '-o', '.', '--num-speakers', '2'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
