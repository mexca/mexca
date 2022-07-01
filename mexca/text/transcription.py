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
        transcription = self._pipeline.transcribe([filepath])

        return transcription[0]


class AudioTextIntegrator:

    def __init__(self, audio_transcriber) -> 'AudioTextIntegrator':
        self._audio_transcriber = audio_transcriber


    def apply(self, filepath, audio_features):
        transcription = self._audio_transcriber.apply(filepath)
        audio_text_features = self.add_transcription(transcription, audio_features)
        return audio_text_features


    def add_transcription(self, transcription, audio_features):
        tokens_text = transcription['transcription'].split(' ')

        time = audio_features['time']

        audio_text_features = audio_features
        audio_text_features['text_token_id'] = np.zeros_like(time)
        audio_text_features['text_token'] = np.full_like(
            time, '', dtype=np.chararray
        )
        audio_text_features['text_token_start'] = np.zeros_like(time)
        audio_text_features['text_token_end'] = np.zeros_like(time)

        char_idx = 0

        for id, token in enumerate(tokens_text):
            start = transcription['start_timestamps'][char_idx] / 1000
            char_idx += len(token)
            end = transcription['end_timestamps'][char_idx-1] / 1000
            #char_idx += 1

            is_token = np.logical_and(
                np.less(time, end), np.greater(time, start)
            )
            audio_text_features['text_token_id'][is_token] = id
            audio_text_features['text_token'][is_token] = token
            audio_text_features['text_token_start'][is_token] = start
            audio_text_features['text_token_end'][is_token] = end

        return audio_text_features


    def _segment_text(self, transcription, annotation):

        text, char_starts, char_ends = transcription['transcription'], transcription['start_timestamps'], transcription['end_timestamps']

        output = {'speech_start':[], 'speech_end':[], 'speaker':[], 'text':[]}

        for speech_turn, _, speaker in annotation.itertracks(yield_label=True):
            # convert pyannote annotation time to ms to match the text transcription
            speech_start, speech_end = 1000 * round(speech_turn.start, 3), 1000 * round(speech_turn.end, 3)

            # extract character within the time window identified in the pyannote annotation (i.e., characters within start and end)
            extracted_text = ''.join(
                [char for i, char in enumerate(text) if (char_ends[i]>=speech_start and char_starts[i]<=speech_end)]
            )

            # append info into a dictionary
            output['speech_start'].append(speech_start)
            output['speech_end'].append(speech_end)
            output['speaker'].append(speaker)
            output['text'].append(extracted_text)

        return output
