"""Transcribe speech from audio to text.
"""

import argparse
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Union
import pyannote.core.json
import stable_whisper
import torch
import whisper
from pyannote.core import Annotation, Segment
from tqdm import tqdm
from whisper.audio import SAMPLE_RATE


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

    def __init__(self, #pylint: disable=too-many-arguments
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

    # These methods must be explictly declared to compare
    # TranscribedSegments with Segments
    def __lt__(self, obj):
        return self.start < obj.start and self.end < obj.end


    def __le__(self, obj):
        return self.start <= obj.start and self.end <= obj.end


    def __gt__(self, obj):
        return self.start > obj.start and self.end > obj.end


    def __ge__(self, obj):
        return self.start >= obj.start and self.end >= obj.end


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
        show_progress: bool, optional, default=True
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

# Adapted from whisper.trascribe.cli
# See: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
def cli():
    """Command line interface for audio transcription.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-a', '--annotation-path', type=str, required=True, dest='annotation_path')
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument("--model", default="small", choices=whisper.available_models())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", type=str, default=None, choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted([k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]))
    parser.add_argument('--sentence-rule', type=str, default=None, dest='sentence_rule')
    parser.add_argument('--show-progress', type=str, default=None, dest='show_progress')

    args = parser.parse_args().__dict__

    transcriber = AudioTranscriber(
        whisper_model=args['model'],
        device=args['device'],
        sentence_rule=args['sentence_rule']
    )

    options = whisper.DecodingOptions(
        language=args['language'],
        without_timestamps=False,
        fp16=torch.cuda.is_available()
    )

    audio_annotation = pyannote.core.json.load_from(args['annotation_path'])

    output = transcriber.apply(
        args['filepath'],
        audio_annotation=audio_annotation,
        options=options,
        show_progress=args['show_progress']
    )

    pyannote.core.json.dump_to(
        output,
        os.path.join(args['outdir'], os.path.basename(args['filepath']) + '.json')
    )


if __name__ == '__main__':
    cli()
