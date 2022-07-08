""" Preprocessing classes and methods """

import os
from moviepy.editor import VideoFileClip
from mexca.core.exceptions import AudioClipError


class Video2AudioConverter(VideoFileClip):
    def __init__(self, filepath) -> 'Video2AudioConverter':
        self.filepath = filepath
        super().__init__(filepath)


    def write_audiofile(self, audio_path, fps=16000):
        if not self.audio:
            video_file = os.path.split(self.filepath)[-1]
            raise AudioClipError(f'Cannot process file "{video_file}" because it does not contain audio')

        self.audio.write_audiofile(audio_path, fps=fps)


    def create_audiofile_path(self):
        audio_path = os.path.splitext(self.filepath)[0] + '.wav'

        return audio_path
