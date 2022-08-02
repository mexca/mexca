""" Audio to text transcription classes and methods """

import numpy as np
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
        transcription = self._pipeline.transcribe([filepath]) # Requires list input!

        return transcription[0] # Output list contains only one element


class AudioTextIntegrator:

    def __init__(self, audio_transcriber) -> 'AudioTextIntegrator':
        self._audio_transcriber = audio_transcriber


    def apply(self, filepath, time):
        """
        Apply audio-to-text transcription pipeline and integrates into a unique output

        Parameters
        -----------------------
        filepath: str,
            audio file path
        time: float,
        verbose: bool,
            Enables the display of a progress bar. Defaul to False.
        """
        transcription = self._audio_transcriber.apply(filepath)
        audio_text_features = self.add_transcription(transcription, time)
        return audio_text_features


    def add_transcription(self, transcription, time):
        tokens_text = transcription['transcription'].split(' ')

        audio_text_features = {
            'text_token_id': np.zeros_like(time),
            'text_token': np.full_like(
                time, '', dtype=np.chararray
            ),
            'text_token_start': np.zeros_like(time),
            'text_token_end': np.zeros_like(time)
        }

        char_idx = 0

        for i, token in enumerate(tokens_text):
            start = transcription['start_timestamps'][char_idx] / 1000
            char_idx += len(token)
            end = transcription['end_timestamps'][char_idx-1] / 1000

            is_token = np.logical_and(
                np.less(time, end), np.greater(time, start)
            )
            audio_text_features['text_token_id'][is_token] = i
            audio_text_features['text_token'][is_token] = token
            audio_text_features['text_token_start'][is_token] = start
            audio_text_features['text_token_end'][is_token] = end

        return audio_text_features
