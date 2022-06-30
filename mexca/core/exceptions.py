""" Create custom exceptions for mexca """

class PipelineError(Exception):
    def __init__(self, message) -> 'PipelineError':
        self.message = message
        super().__init__(message)


class AudioClipError(Exception):
    def __init__(self, message) -> 'AudioClipError':
        self.message = message
        super().__init__(message)


class ModelTranscriberInitError(Exception):
    def __init__(self, message) -> 'ModelTranscriberInitError':
        self.message = message
        super().__init__(message)
