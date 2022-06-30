""" Pipeline class and methods """

import os
from mexca.core.exceptions import PipelineError
from mexca.core.output import Multimodal
from mexca.core.preprocessing import Video2AudioConverter

class Pipeline:
    def __init__(self, video=None, audio=None, text=None) -> 'Pipeline':
        if text and not audio:
            raise PipelineError('Cannot initialize a "text" component if no "audio" component was specified')
        self.video = video
        self.audio = audio
        self.text = text


    def apply(self, filepath, remove_audiofile=False) -> 'Multimodal':
        if self.video:
            video_result = self.video.apply(filepath)
        else:
            video_result = None

        if self.audio:
            with Video2AudioConverter(filepath) as clip:
                audio_path = clip.create_audiofile_path()
                clip.write_audiofile(audio_path)

            audio_result = self.audio.apply(audio_path)

            if self.text and audio_result:
                text_result = self.text.apply(audio_path, audio_result)
            else:
                text_result = None

            if remove_audiofile:
                os.remove(audio_path)

        else:
            audio_result = None

        pipeline_result = Multimodal(video_result, audio_result, text_result)

        return pipeline_result
