""" Test preprocessing classes and methods """

import os
import pytest
from mexca.core.exceptions import AudioClipError
from mexca.core.preprocessing import Video2AudioConverter

class TestVideo2AudioConverter:
    filepath = os.path.join('tests', 'test_files', 'test_video_multi_5_frames.mp4')
    converter = Video2AudioConverter(filepath)
    reference_audio_path = os.path.join('tests', 'test_files', 'test_video_multi_5_frames.wav')

    @pytest.mark.xfail(raises=AudioClipError)
    def test_write_audiofile(self):
        with self.converter as clip:
            clip.write_audiofile(self.reference_audio_path)


    def test_create_audio_file_path(self):
        assert self.converter.create_audiofile_path() == self.reference_audio_path
