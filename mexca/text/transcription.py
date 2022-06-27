""" Audio to text transcription classes and methods """

from huggingsound import SpeechRecognitionModel
from mexca.core.exceptions import ModelTranscriberInitError

class AudioTranscriber:

    def __init__(self,language=None) -> 'AudioTranscriber':
        self.language = language

        if language == 'dutch':
          self._pipeline = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-dutch")
        elif language == 'english':
          self._pipeline = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        else:
          raise ModelTranscriberInitError("Invalid language. Please specify either 'dutch' or 'english'")

    def apply(self, filepath):

        transcription = self._pipeline.transcribe(filepath)

        return transcription[0]
