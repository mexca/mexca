""" Test speaker identification classes and methods """

import os
import subprocess
from pyannote.core import Annotation, Segment
from mexca.audio import SpeakerIdentifier


class TestSpeakerIdentifier:
    use_auth_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else True
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )


    def test_apply(self):
        speaker_identifier = SpeakerIdentifier(use_auth_token=self.use_auth_token)
        annotation = speaker_identifier.apply(self.filepath)

        assert isinstance(annotation, Annotation)
        
        for seg, trk, spk in annotation.itertracks(yield_label=True):
            assert isinstance(seg, Segment)
            assert isinstance(seg.start, float)
            assert 5.0 >= seg.start >= 0.0
            assert isinstance(seg.end, float)
            assert 5.0 >= seg.end >= 0.0
            assert isinstance(trk, int)
            assert isinstance(spk, int)


    def test_apply_num_speakers(self):
        num_speakers = 2
        speaker_identifier = SpeakerIdentifier(
            num_speakers=num_speakers,
            use_auth_token=self.use_auth_token,
        )
        annotation = speaker_identifier.apply(self.filepath)

        assert isinstance(annotation, Annotation)
        
        for seg, trk, spk in annotation.itertracks(yield_label=True):
            assert isinstance(seg, Segment)
            assert isinstance(seg.start, float)
            assert 5.0 >= seg.start >= 0.0
            assert isinstance(seg.end, float)
            assert 5.0 >= seg.end >= 0.0
            assert isinstance(trk, int)
            assert isinstance(spk, int)
            assert spk <= num_speakers


    def test_cli(self):
        out_filename = os.path.basename(self.filepath) + '.rttm'
        subprocess.run(['identify-speakers', '-f', self.filepath,
                        '-o', '.', '--num-speakers', '2'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)