""" Audio to text transcription classes and methods """

from huggingsound import SpeechRecognitionModel
from mexca.core.exceptions import ModelTranscriberInitError
from mexca.audio.speaker_id import SpeakerIdentifier

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

    def __init__(self, audio_transcriber, speaker_identifier) -> 'AudioTextIntegrator':
        self._audio_transcriber = audio_transcriber
        self._speech_segmenter = speaker_identifier


    def apply(self, filepath):
        transcription = self._audio_transcriber.apply(filepath)
        annotation = self._speech_segmenter.apply(filepath)
        out = self._segment_text(transcription, annotation)
        return out


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
