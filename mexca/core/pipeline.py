""" Pipeline class and methods """

from mexca.core.output import Multimodal

class Pipeline:
    enabled_video: bool
    enabled_audio: bool
    enabled_text: bool

    filepath = None

    _video_class = None
    _audio_class = None
    _text_class = None

    def __init__(self,
        enabled_video,
        enabled_audio,
        enabled_text,
    ) -> 'Pipeline':
        self.enabled_video = enabled_video
        self.enabled_audio = enabled_audio
        self.enabled_text = enabled_text

        if self.get_enabled_video() and self._video_class is not None:
            self.video = self._video_class()

        if self.get_enabled_audio() and self._audio_class is not None:
            self.audio = self._audio_class()

        if self.get_enabled_text() and self._text_class is not None:
            self.text = self._text_class()


    def get_enabled_video(self) -> bool:
        return self.enabled_video


    def get_enabled_audio(self) -> bool:
        return self.enabled_audio


    def get_enabled_text(self) -> bool:
        return self.enabled_text


    def set_filepath(self, filepath) -> None:
        self.filepath = filepath


    def get_filepath(self) -> str:
        if self.filepath is None:
            raise AttributeError('Attribute "filepath" must be set before it can be accessed')
        return self.filepath


    def apply(self, filepath) -> 'Mexca':

        if self.get_enabled_video():
            video_result = self.video.apply(filepath)
        else:
            video_result = None

        if self.get_enabled_audio():
            audio_result = self.audio.apply(filepath)
        else:
            audio_result = None

        if self.get_enabled_text():
            text_result = self.text.apply(filepath)
        else:
            text_result = None

        pipeline_result = Multimodal(video_result, audio_result, text_result)

        return pipeline_result
