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


class SkipFramesError(Exception):
    """More frames are skipped than exist in a video.
    """
    def __init__(self, message) -> 'SkipFramesError':
        self.message = message
        super().__init__(message)
