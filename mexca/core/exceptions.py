""" Create custom exceptions for mexca """

class PipelineError(Exception):
    def __init__(self, message) -> 'PipelineError':
        self.message = message
        super().__init__(self.message)


class ModelTranscriberInitError(Exception):
    def __init__(self, message) -> 'ModelTranscriberInitError':
        self.message = message
        super().__init__(self.message)
