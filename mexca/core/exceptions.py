""" Create custom exceptions for mexca """


class PipelineError(Exception):
    def __init__(self, message) -> 'PipelineError':
        self.message = message
        super().__init__(message)


class AudioClipError(Exception):
    def __init__(self, message) -> 'AudioClipError':
        self.message = message
        super().__init__(message)


class VoiceFeatureError(Exception):
    def __init__(self, message) -> 'VoiceFeatureError':
        self.message = message
        super().__init__(message)


class TimeStepError(Exception):
    def __init__(self) -> 'TimeStepError':
        self.message = 'To extract audio features, video processing must be enabled or the argument "time_step" must be supplied'
        super().__init__(self.message)


class TimeStepWarning(Warning):
    def __init__(self, message) -> 'TimeStepWarning':
        self.message = message
        super().__init__(message)


class ModelTranscriberInitError(Exception):
    def __init__(self, message) -> 'ModelTranscriberInitError':
        self.message = message
        super().__init__(message)
