"""Transcribe speech from audio to text.
"""

import os
import numpy as np
from huggingsound import SpeechRecognitionModel
from parselmouth import Sound
from mexca.core.exceptions import TimeStepError
from mexca.core.utils import create_time_var_from_step


class AudioTranscriber:
    """Transcribe speech from audio to text.

    Parameters
        ----------
        language: {'english', 'dutch'}
            The name of the language that is transcribed from the audio file.
            Currently, only English and Dutch are available.

    """
    def __init__(self, language) -> 'AudioTranscriber':
        self.language = language

        if self.language == 'dutch':
            self.hugging_sound = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-dutch")
        else:
            self.hugging_sound = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")


    @property
    def language(self):
        return self._language


    @language.setter
    def language(self, new_language):
        if isinstance(new_language, str):
            if new_language.lower() in ('english', 'dutch'):
                self._language = new_language.lower()
            else:
                raise ValueError('Please specify a valid, available language, either "english" or "dutch"')
        else:
            raise ValueError('Can only set "language" to str')


    @property
    def hugging_sound(self):
        return self._hugging_sound


    @hugging_sound.setter
    def hugging_sound(self, new_hugging_sound):
        if isinstance(new_hugging_sound, SpeechRecognitionModel):
            self._hugging_sound = new_hugging_sound
        else:
            raise ValueError('Can only set "hugging_sound" to "SpeechRecognitionModel"')


    def apply(self, filepath):
        """Transcribe speech in an audio file to text.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.

        Returns
        -------
        dict
            A dictionary with key-value pairs:
            - `transcription`: A string with the audio transcription.
            - `start_timestamps`: A list of ints containing the start times of each character (in ms).
            - `end_timestamps`: A list of ints containing the end times of each character (in ms).
            - `probabilities`: A list of floats containing the probabilities of each character.

        """
        transcription = self.hugging_sound.transcribe([filepath]) # Requires list input!

        return transcription[0] # Output list contains only one element


class AudioTextIntegrator:
    """Integrate audio transcription and audio features.

    Parameters
    ----------
    audio_transcriber: AudioTranscriber
        An instance of the ``AudioTranscriber`` class.
    time_step: float or None, default=None
        The interval at which transcribed text is matched to audio frames.
        Only used if the ``apply`` method has `time=None`.

    """
    def __init__(self, audio_transcriber, time_step=None) -> 'AudioTextIntegrator':
        self.audio_transcriber = audio_transcriber
        self.time_step = time_step


    @property
    def audio_transcriber(self):
        return self._audio_transcriber


    @audio_transcriber.setter
    def audio_transcriber(self, new_audio_transcriber):
        if isinstance(new_audio_transcriber, AudioTranscriber):
            self._audio_transcriber = new_audio_transcriber
        else:
            raise ValueError('Can only set "audio_transcriber" to instance of AudioTranscriber class')


    @property
    def time_step(self):
        return self._time_step


    @time_step.setter
    def time_step(self, new_time_step):
        if new_time_step:
            if isinstance(new_time_step, (float, int)):
                if new_time_step >= 0.0:
                    self._time_step = new_time_step
                else:
                    raise ValueError('Can only set "time_step" to values >= zero')
            else:
                raise ValueError('Can only set "time_step" to float, int, or None')
        else:
            self._time_step = new_time_step


    def apply(self, filepath, time):
        """
        Integrate audio transcription and audio features.

        Parameters
        -----------------------
        filepath: str or path
            Path to the audio file.
        time: list or numpy.ndarray or None
            A list of floats or array containing time points to which the transcribed text is matched.

        Returns
        -------
        dict
            A dictionary with audio and text features.
            See the ``add_transcription`` method for details.

        """
        if not os.path.exists(filepath):
            raise ValueError('Argument "filepath" must be str or path')

        if time and not isinstance(time, (list, np.ndarray)):
            raise ValueError('Argument "time" must be list or numpy.ndarray')

        transcription = self.audio_transcriber.apply(filepath)

        snd = Sound(filepath)

        if not time and not self.time_step:
            raise TimeStepError()

        if not time:
            end_time = snd.get_end_time()
            time = create_time_var_from_step(self.time_step, end_time)

        audio_text_features = self.add_transcription(transcription, time)
        return audio_text_features


    def add_transcription(self, transcription, time):
        """Add audio transcription to audio features.

        Parameters
        ----------
        transcription: dict
            The result of ``AudioTranscriber.apply``.
        time: list or numpy.ndarray
            A list of floats or array containing time points to which the transcribed text is matched.

        Returns
        -------
        dict
            A dictionary with key-value pairs:
            - `text_token_id`: An array indexing the token (word) in the transcription.
            - `text_token`: An array with the tokens of the transcription
            - `text_token_start`: An array with the start times of tokens.
            - `text_token_end`: An array with the end times of tokens.

        """
        # Split transcription into words (tokens)
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

            # Index time points that include token
            is_token = np.logical_and(
                np.less(time, end), np.greater(time, start)
            )
            audio_text_features['text_token_id'][is_token] = i
            audio_text_features['text_token'][is_token] = token
            audio_text_features['text_token_start'][is_token] = start
            audio_text_features['text_token_end'][is_token] = end

        return audio_text_features
