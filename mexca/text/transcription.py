"""Transcribe speech from audio to text.
"""

import numpy as np
from deepmultilingualpunctuation import PunctuationModel
from huggingsound import SpeechRecognitionModel
from parselmouth import Sound
from spacy.language import Language
from spacy.tokens import Token
from spacy.vocab import Vocab
from mexca.core.exceptions import TimeStepError
from mexca.core.utils import create_time_var_from_step
from mexca.text.sentiment import SentimentExtractor



class AudioTranscriber:
    """Transcribe speech from audio to text.

    Parameters
    ----------
    language: {'english', 'dutch'}
        The name of the language that is transcribed from the audio file.
        Currently, only English and Dutch are available.

    Attributes
    ----------
    hugging_sound

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
            raise TypeError('Can only set "language" to str')


    @property
    def hugging_sound(self):
        """The HuggingSound model for speech recognition. Must be instance of `SpeechRecognitionModel` class.
        See `huggingsound <https://github.com/jonatasgrosman/huggingsound>`_ for details.
        """
        return self._hugging_sound


    @hugging_sound.setter
    def hugging_sound(self, new_hugging_sound):
        if isinstance(new_hugging_sound, SpeechRecognitionModel):
            self._hugging_sound = new_hugging_sound
        else:
            raise TypeError('Can only set "hugging_sound" to "SpeechRecognitionModel"')


    def apply(self, filepath):
        """Transcribe speech in an audio file to text.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.

        Returns
        -------
        dict
            A dictionary with extracted text features.

        """
        transcription = self.hugging_sound.transcribe([filepath]) # Requires list input!

        return transcription[0] # Output list contains only one element


class TextRestaurator:
    """Restore punctuation and sentence structures in text.

    Parameters
    ----------
    model: str or None, default=None
        The name of the punctuation model (e.g., on HuggingFace Hub).

    """
    def __init__(self, model=None) -> 'TextRestaurator':
        if model:
            self.model = model
        else:
            self.model = 'oliverguhr/fullstop-punctuation-multilang-large'
        self.punctuator = PunctuationModel(self.model)
        self.sentencizer = Language(Vocab())
        self.sentencizer.add_pipe('sentencizer')


    @property
    def model(self):
        return self._model


    @model.setter
    def model(self, new_model):
        if isinstance(new_model, str):
            self._model = new_model
        else:
            raise TypeError('Can only set "model" to str')


    @property
    def punctuator(self):
        return self._punctuator


    @punctuator.setter
    def punctuator(self, new_punctuator):
        if isinstance(new_punctuator, PunctuationModel):
            self._punctuator = new_punctuator
        else:
            raise TypeError('Can only set "punctuator" to instance of "PunctuationModel" class')


    @property
    def sentencizer(self):
        return self._sentencizer


    @sentencizer.setter
    def sentencizer(self, new_sentencizer):
        if isinstance(new_sentencizer, Language):
            self._sentencizer = new_sentencizer
        else:
            raise TypeError('Can only set "sentencizer" to instance of "Language" class')


    @staticmethod
    def set_token_extensions():
        """Add extensions for SpaCy tokens.

        Sets custom token attributes `time_start` and `time_end`.

        """
        Token.set_extension('time_start')
        Token.set_extension('time_end')


    def apply(self, text):
        """Restore punctuation and sentence structures in text.

        Adds five different punctuation characters to the text: '.', ',', '?', '-', ':'.
        Converts the text into a ``Doc`` object with a `sents` attribute containing the
        sentences of the text. See the `spacy https://spacy.io/api/sentencizer`_ for details.

        Parameters
        ----------
        text: str or list[str]
            Text strings to be restored.

        Returns
        -------
        spacy.tokens.Doc
            The restored text in a ``Doc`` class instance.

        """
        restored_text = self.punctuator.restore_punctuation(text['transcription'])
        restored_docs = self.sentencizer(restored_text)

        char_idx = 0

        for token in restored_docs:
            token._.time_start = float(text['start_timestamps'][char_idx] / 1000)
            char_idx += len(token)
            token._.time_end = float(text['end_timestamps'][char_idx-1] / 1000)

        return restored_docs


class AudioTextIntegrator:
    """Integrate audio transcription and audio features.

    Parameters
    ----------
    audio_transcriber: AudioTranscriber
        An instance of the `AudioTranscriber` class.
    text_restaurator: TextRestaurator
        An instance of the `TextRestaurator` class.
    sentiment_extractor: SentimentExtractor
        An instance of the `SentimentExtractor` class.
    time_step: float or None, default=None
        The interval at which transcribed text is matched to audio frames.
        Only used if the `apply` method has `time=None`.

    """
    def __init__(self, audio_transcriber, text_restaurator, sentiment_extractor, time_step=None) -> 'AudioTextIntegrator':
        self.audio_transcriber = audio_transcriber
        self.text_restaurator = text_restaurator
        self.sentiment_extractor = sentiment_extractor
        self.time_step = time_step


    @property
    def audio_transcriber(self):
        return self._audio_transcriber


    @audio_transcriber.setter
    def audio_transcriber(self, new_audio_transcriber):
        if isinstance(new_audio_transcriber, AudioTranscriber):
            self._audio_transcriber = new_audio_transcriber
        else:
            raise TypeError('Can only set "audio_transcriber" to instance of "AudioTranscriber" class')


    @property
    def text_restaurator(self):
        return self._text_restaurator


    @text_restaurator.setter
    def text_restaurator(self, new_text_restaurator):
        if isinstance(new_text_restaurator, TextRestaurator):
            self._text_restaurator = new_text_restaurator
        else:
            raise TypeError('Can only set "text_restaurator" to instance of "TextRestaurator" class')


    @property
    def sentiment_extractor(self):
        return self._sentiment_extractor


    @sentiment_extractor.setter
    def sentiment_extractor(self, new_sentiment_extractor):
        if isinstance(new_sentiment_extractor, SentimentExtractor):
            self._sentiment_extractor = new_sentiment_extractor
        else:
            raise TypeError('Can only set "sentiment_extractor" to instance of "SentimentExtractor" class')


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
                raise TypeError('Can only set "time_step" to float, int, or None')
        else:
            self._time_step = new_time_step


    def apply(self, filepath, time):
        """
        Integrate audio transcription and audio features.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.
        time: list or numpy.ndarray or None
            A list of floats or array containing time points to which the transcribed text is matched.

        Returns
        -------
        dict
            A dictionary with audio and text features.
            See the `add_transcription` method for details.

        """
        if time and not isinstance(time, (list, np.ndarray)):
            raise TypeError('Argument "time" must be list or numpy.ndarray')

        snd = Sound(filepath)

        if not time and not self.time_step:
            raise TimeStepError()

        if not time:
            end_time = snd.get_end_time()
            time = create_time_var_from_step(self.time_step, end_time)

        transcription = self.audio_transcriber.apply(filepath)
        docs = self.text_restaurator.apply(transcription)
        sentiment = self.sentiment_extractor.apply(docs)

        audio_text_features = self.integrate(time, docs, sentiment)

        return audio_text_features


    def integrate(self, time, docs, sentiment):
        """Integrate audio transcription with audio features and text sentiment.

        Parameters
        ----------
        time: list or numpy.ndarray
            A list of floats or array containing time points to which the transcribed text is matched.
        docs: spacy.tokens.Doc
            A ``Doc`` instance containing the audio transcription.
        sentiment: list or np.ndarray
            List or array containing sentiment probabilities with shape N x 3 where N is the number of sentences in the text.
            The order of sentiments must be negative, neutral, positive.

        Returns
        -------
        dict
            A dictionary with extracted text features.

        """

        audio_text_features = {
            'time': time,
            'text_token_id': np.full_like(time, np.nan),
            'text_token': np.full_like(time, np.nan, dtype=np.chararray),
            'text_token_start': np.full_like(time, np.nan),
            'text_token_end': np.full_like(time, np.nan),
            'text_sent_id': np.full_like(time, np.nan),
            'text_sent_pos': np.full_like(time, np.nan),
            'text_sent_neg': np.full_like(time, np.nan),
            'text_sent_neu': np.full_like(time, np.nan)
        }

        for i, sent in enumerate(docs.sents):
            for j, token in enumerate(sent):
                # Index time points that include token
                is_token = np.logical_and(
                    np.less(time, token._.time_end), np.greater(time, token._.time_start)
                )
                audio_text_features['text_token_id'][is_token] = j
                audio_text_features['text_token'][is_token] = token.text
                audio_text_features['text_token_start'][is_token] = token._.time_start
                audio_text_features['text_token_end'][is_token] = token._.time_end
                audio_text_features['text_sent_id'][is_token] = i
                audio_text_features['text_sent_pos'][is_token] = sentiment[i][2]
                audio_text_features['text_sent_neg'][is_token] = sentiment[i][0]
                audio_text_features['text_sent_neu'][is_token] = sentiment[i][1]

        return audio_text_features
