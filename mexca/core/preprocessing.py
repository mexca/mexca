"""Preprocessing classes and methods.
"""

import os
from moviepy.editor import VideoFileClip
from mexca.core.exceptions import AudioClipError


class Video2AudioConverter(VideoFileClip):
    """Write the audio signal from a video to a (temporary) file.
    """
    def __init__(self, filepath, **kwargs) -> 'Video2AudioConverter':
        """Create a class to write the audio signal from a video to a file.

        Parameters
        ----------
        filepath: str or path
            Path to the video file.
        **kwargs: dict, optional
            Additional arguments to construct the ```moviepy.editor.VideoFileClip`` class.

        Returns
        -------
        A ``Video2AudioConverter`` class instance.

        """
        self.filepath = filepath
        super().__init__(filepath, **kwargs)


    def write_audiofile(self, audio_path, fps=16000):
        """Write an audio signal to a file.

        Parameters
        ----------
        audio_path: str or path
            Path to the audio file. Should specify a file format (e.g., '.wav').
        fps: int, default=16000
            Frames per second of the written audio file.
            This should be 16000 to align with the audio feature extraction.

        """
        if not self.audio:
            video_file = os.path.split(self.filepath)[-1]
            raise AudioClipError(f'Cannot process file "{video_file}" because it does not contain audio')

        self.audio.write_audiofile(audio_path, fps=fps, logger=None)


    def create_audiofile_path(self):
        """Create a path for the audio file.

        This function is intended to conveniently create a path to a temporary audio file.

        Returns
        -------
        path
            A path to the audio file matching the one of the video file, except that the file format is '.wav'.

        """
        audio_path = os.path.splitext(self.filepath)[0] + '.wav'

        return audio_path
