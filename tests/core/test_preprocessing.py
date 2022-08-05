""" Test preprocessing classes and methods """

import os
import pytest
from mexca.core.exceptions import AudioClipError
from mexca.core.preprocessing import Video2AudioConverter


class TestVideo2AudioConverter:
    filepath = os.path.join('tests', 'test_files', 'test_video_audio_5_seconds.mp4')
    converter = Video2AudioConverter(filepath)
    converter_no_audio = Video2AudioConverter(filepath, audio=False)
    reference_audio_path = os.path.join('tests', 'test_files', 'test_video_audio_5_seconds.wav')

    def test_write_audiofile(self):
        with self.converter as clip:
            clip.write_audiofile(self.reference_audio_path)

        assert os.path.exists(self.reference_audio_path)
        os.remove(self.reference_audio_path) # Remove audio file


    def test_write_audiofile_error(self):
        with pytest.raises(expected_exception=AudioClipError):
            with self.converter_no_audio as clip:
                clip.write_audiofile(self.reference_audio_path)


    def test_create_audio_file_path(self):
        assert self.converter.create_audiofile_path() == self.reference_audio_path
