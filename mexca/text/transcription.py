"""Transcribe speech from audio to text.
"""

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import stable_whisper
import torch
import whisper
from parselmouth import Sound
from pyannote.core import Annotation, Segment
from tqdm import tqdm
from whisper.audio import SAMPLE_RATE
from mexca.core.exceptions import TimeStepError
from mexca.core.utils import create_time_var_from_step
from mexca.text.sentiment import SentimentExtractor


@dataclass
class Sentence:
    """Annotate a sentence within a transcribed speech segment.

    Parameters
    ----------
    text: str
        The transcribed sentence.
    start, end: float
        The start and end of the sentence within in the segment (in seconds).
    sent_pos, sent_neg, sent_neu: float, optional
        The positive, negative, and neutral sentiment scores of the sentence.

    """
    text: str
    start: float
    end: float
    sent_pos: Optional[float] = None
    sent_neg: Optional[float] = None
    sent_neu: Optional[float] = None


class TranscribedSegment(Segment):
    """Annotate an audio segment with transcribed text and sentiment.

    Parameters
    ----------
    start, end: float
        The start and end of the segment (in seconds).
    text: str, optional
        The transcribed text of speech within the segment.
    lang: str, optional
        The detected language of speech within the segment.
    sents: list, optional
        A list of sentences in the transcribed text.

    """

    def __init__(self, # pylint: disable=too-many-arguments
        start: float,
        end: float,
        text: Optional[str] = None,
        lang: Optional[str] = None,
        sents: Optional[List[Sentence]] = None
    ):
        super().__init__(start, end)
        self.text = text
        self.lang = lang
        self.sents = sents


class AudioTranscriber:
    """Transcribe speech from audio to text.

    Parameters
    ----------
    whisper_model: str, optional, default='small'
        The name of the whisper model that is used for transcription. Available models are
        `['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']`.
    device: str or torch.device, optional, default='cpu'
        The name of the device onto which the whisper model should be loaded and run. If CUDA support is
        available, this can be `'cuda'`, otherwise use `'cpu'` (the default).
    sentence_rule: str, optional
        A regular expression used to split segment transcripts into sentences. If `None` (default), it splits
        the text at all '.', '?', '!', and ':' characters that are followed by whitespace characters. It
        omits single or multiple words abbreviated with dots (e.g., 'Nr. ' and 'e.g. ').

    Attributes
    ----------
    transcriber: whisper.Whisper
        The loaded whisper model for audio transcription.

    """
    def __init__(self,
        whisper_model: Optional[str] = 'small',
        device: Optional[Union[str, torch.device]] = 'cpu',
        sentence_rule: Optional[str] = None
    ):
        self.whisper_model = whisper_model
        self.device = device
        self.transcriber = stable_whisper.load_model(whisper_model, device)

        if not sentence_rule:
            self.sentence_rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|:)\s"
        else:
            self.sentence_rule = sentence_rule


    def apply(self, # pylint: disable=too-many-locals
        filepath: str,
        audio_annotation: Annotation,
        options: Optional[whisper.DecodingOptions] = None,
        show_progress: bool = True
    ) -> Annotation:
        """Transcribe speech in an audio file to text.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        audio_annotation: pyannote.core.Annotation
            The audio annotation object returned by the pyannote.audio speaker diarization pipeline.
        options: whisper.DecodingOptions, optional
            Options for transcribing the audio file. If `None`, transcription is done without timestamps,
            and with a number format that depends on whether CUDA is available:
            FP16 (half-precision floating points) if available,
            FP32 (single-precision floating points) otherwise.
        show_progress: bool
            Whether a progress bar is displayed or not.

        Returns
        -------
        pyannote.core.Annotation
            An annotation object containing segments with transcription.

        """
        if not options:
            options = self.get_default_options()

        audio = torch.Tensor(whisper.load_audio(filepath))

        new_annotation = Annotation()

        for seg, trk, spk in tqdm( # segment, track, speaker
            audio_annotation.itertracks(yield_label=True),
            total=len(audio_annotation),
            disable=not show_progress
        ):
            # Get start and end frame
            start = int(seg.start * SAMPLE_RATE)
            end = int(seg.end * SAMPLE_RATE)

            # Subset audio signal
            audio_sub = audio[start:end]

            output = self.transcriber.transcribe(audio_sub, **asdict(options))

            segment_text = output['text'].strip()

            # Split text into sentences
            sents = re.split(self.sentence_rule, segment_text)


            def annotate_sentences(sents, output, seg_start):
                """Helper function to annotate sentence start and end timestamps.
                """
                # Concatenate word timestamps from every segment
                whole_word_timestamps = []

                for segment in output['segments']:
                    whole_word_timestamps.extend(segment['whole_word_timestamps'])

                # Get sentence start and end timestamps
                sents_ts = []

                # If word-level timestamps are available
                if len(whole_word_timestamps) > 0:
                    idx = 0

                    for sent in sents:
                        sent_len = len(sent.split(" ")) - 1

                        start = whole_word_timestamps[idx]['timestamp']
                        end = whole_word_timestamps[idx + sent_len]['timestamp']

                        sents_ts.append(Sentence(sent, seg_start + start, seg_start + end))

                        idx += sent_len + 1

                return sents_ts


            sents_ts = annotate_sentences(sents, output, seg.start)

            # Add new segment with text
            new_seg = TranscribedSegment(
                seg.start,
                seg.end,
                segment_text,
                output['language'],
                sents = sents_ts
            )

            new_annotation[new_seg, trk] = spk

        return new_annotation


    @staticmethod
    def get_default_options() -> whisper.DecodingOptions:
        """Set default options for transcription.

        Returns
        -------
        whisper.DecodingOptions

        """
        return whisper.DecodingOptions(
            without_timestamps=False,
            fp16=torch.cuda.is_available()
        )


class AudioTextIntegrator:
    """Integrate audio transcription and audio features.

    Parameters
    ----------
    audio_transcriber: AudioTranscriber
        A class instance for audio transcription.
    sentiment_extractor: SentimentExtractor
        A class instance for sentiment prediction.
    time_step: float, optional
        The interval at which transcribed text is matched to audio frames.
        Must be > 0. Only used if the `apply` method has `time=None`.

    """
    def __init__(self,
        audio_transcriber: AudioTranscriber,
        sentiment_extractor: SentimentExtractor,
        time_step: Optional[float] = None
    ):
        self.audio_transcriber = audio_transcriber
        self.sentiment_extractor = sentiment_extractor
        self.time_step = time_step


    @property
    def time_step(self) -> float:
        return self._time_step


    @time_step.setter
    def time_step(self, new_time_step: float):
        if new_time_step:
            if new_time_step > 0.0:
                self._time_step = new_time_step
            else:
                raise ValueError('Can only set "time_step" to values > zero')

        self._time_step = new_time_step


    def apply(self, # pylint: disable=too-many-arguments
        filepath: str,
        audio_annotation: Annotation,
        time: Optional[Union[List[float], np.ndarray]] = None,
        options: Optional[whisper.DecodingOptions] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Integrate audio transcription and audio features.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        audio_annotation: pyannote.core.Annotation
            An annotation object containing speech segments.
        time: list or numpy.ndarray, optional
            A list of floats or array containing time points to which the transcribed text is matched.
            Is only optional if `AudioTextIntegrator.time_step` is not `None`.
        options: whisper.DecodingOptions, optional
            Options for transcribing the audio file. If `None`, transcription is done without timestamps,
            and with a number format that depends on whether CUDA is available:
            FP16 (half-precision floating points) if available,
            FP32 (single-precision floating points) otherwise.
        show_progress: bool
            Whether a progress bar is displayed or not.

        Returns
        -------
        dict
            A dictionary with text features matched to audio features. Text features are 'segment_text',
            'segment_sent_pos', 'segment_sent_neg', and 'segment_sent_neu'. They contain the segment
            transcriptions as well as positive, negative, and neutral sentiment scores.

        """
        if time and not isinstance(time, (list, np.ndarray)):
            raise TypeError('Argument "time" must be list or numpy.ndarray')

        snd = Sound(filepath)

        if not time and not self.time_step:
            raise TimeStepError()

        if not time:
            end_time = snd.get_end_time()
            time = create_time_var_from_step(self.time_step, end_time)

        transcription = self.audio_transcriber.apply(filepath, audio_annotation, options, show_progress)
        transcription = self.sentiment_extractor.apply(transcription)

        audio_text_features = {
            'time': time,
            'segment_text': np.full_like(time, np.nan, dtype=np.chararray),
            'segment_lang': np.full_like(time, np.nan, dtype=np.chararray),
            'sentence_id': np.full_like(time, np.nan),
            'sentence_start': np.full_like(time, np.nan),
            'sentence_end': np.full_like(time, np.nan),
            'sentence_text': np.full_like(time, np.nan, dtype=np.chararray),
            'sentence_sent_pos': np.full_like(time, np.nan),
            'sentence_sent_neg': np.full_like(time, np.nan),
            'sentence_sent_neu': np.full_like(time, np.nan)
        }

        for seg in transcription.itersegments():
            is_segment = np.logical_and(
                np.less(time, seg.end), np.greater(time, seg.start)
            )

            audio_text_features['segment_text'][is_segment] = seg.text
            audio_text_features['segment_lang'][is_segment] = seg.lang

            for i, sent in enumerate(seg.sents):
                is_sent = np.logical_and(
                    np.less(time, sent.end), np.greater(time, sent.start)
                )

                audio_text_features['sentence_id'][is_sent] = i
                audio_text_features['sentence_start'][is_sent] = sent.start
                audio_text_features['sentence_end'][is_sent] = sent.end
                audio_text_features['sentence_text'][is_sent] = sent.text
                audio_text_features['sentence_sent_pos'][is_sent] = sent.sent_pos
                audio_text_features['sentence_sent_neg'][is_sent] = sent.sent_neg
                audio_text_features['sentence_sent_neu'][is_sent] = sent.sent_neu

        return audio_text_features
