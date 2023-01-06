""" Test speaker identification classes and methods """

import os
import subprocess
from mexca.audio import SpeakerIdentifier
from mexca.data import RttmAnnotation, RttmSegment


class TestSpeakerIdentifier:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )


    @staticmethod
    def check_rttm_annotation(annotation):
        assert isinstance(annotation, RttmAnnotation)
        
        for seg in annotation.segments:
            assert isinstance(seg, RttmSegment)
            assert isinstance(seg.tbeg, float)
            assert 5.0 >= seg.tbeg >= 0.0
            assert isinstance(seg.tdur, float)
            assert 5.0 >= seg.tdur >= 0.0
            assert isinstance(seg.name, int)


    def test_apply(self):
        speaker_identifier = SpeakerIdentifier(use_auth_token=self.use_auth_token)
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
        out_filename = os.path.basename(self.filepath) + '.rttm'
        subprocess.run(['identify-speakers', '-f', self.filepath,
                        '-o', '.', '--num-speakers', '2'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)