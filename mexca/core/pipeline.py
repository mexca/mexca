""" Pipeline class and methods """

from mexca.core.exceptions import PipelineError
from mexca.core.output import Multimodal

class Pipeline:
    def __init__(self, video=None, audio=None, text=None) -> 'Pipeline':
        if text and not audio:
            raise PipelineError('Cannot initialize a "text" component when no "audio" component was specified')
        self.video = video
        self.audio = audio
        self.text = text


    def apply(self, filepath) -> 'Multimodal':
        if self.video:
            video_result = self.video.apply(filepath)
        else:
            video_result = None

        if self.audio:
            audio_result = self.audio.apply(filepath)
        else:
            audio_result = None

        if self.text:
            text_result = self.text.apply(filepath)
        else:
            text_result = None

        pipeline_result = Multimodal(video_result, audio_result, text_result)

        return pipeline_result
