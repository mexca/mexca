"""Custom exceptions for mexca.
"""

class AudioClipError(Exception):
    """The pipeline attempts to process audio for a video file without audio signal.
    """
    def __init__(self, message) -> 'AudioClipError':
        self.message = message
        super().__init__(message)


class TimeStepError(Exception):
    """Video processing is disabled and no `time_step` argument is supplied.
    """
    def __init__(self) -> 'TimeStepError':
        self.message = 'To extract audio features, video processing must be enabled or the argument "time_step" must be supplied'
        super().__init__(self.message)


class TimeStepWarning(Warning):
    """The `time_step` argument does not match the video/audio length.
    """
    def __init__(self, message) -> 'TimeStepWarning':
        self.message = message
        super().__init__(message)


class ModelTranscriberInitError(Exception):
    """A language name that is not available is chosen for transcription.
    """
    def __init__(self, message) -> 'ModelTranscriberInitError':
        self.message = message
        super().__init__(message)
