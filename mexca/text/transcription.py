"""Transcribe speech from audio to text.
"""

import argparse
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import List, Optional, Union
import srt
import stable_whisper
import torch
import whisper
from tqdm import tqdm
from whisper.audio import SAMPLE_RATE


@dataclass
class RttmSegment:
    type: str
    file: str
    chnl: int
    tbeg: float
    tdur: float
    ortho: Optional[str] = None
    stype: Optional[str] = None
    name: Optional[str] = None
    conf: Optional[float] = None


def _get_rttm_header() -> List[str]:
    return ["type", "file", "chnl", "tbeg", 
            "tdur", "ortho", "stype", "name", 
            "conf"]


@dataclass
class RttmAnnotation:
    segments: List[RttmSegment]
    header: List[str] = field(default_factory=_get_rttm_header)


    @classmethod
    def from_pyannote(cls, annotation: 'pyannote.core.Annotation'):
        segments = []

        for seg, _, spk in annotation.itertracks(yield_label=True):
            segments.append(RttmSegment(
                type='SPEAKER',
                file=annotation.uri,
                chnl=1,
                tbeg=seg.start,
                tdur=seg.duration,
                name=spk
            ))

        return cls(segments)


    @classmethod
    def from_rttm(cls, filename: str):
        with open(filename, "r", encoding='utf-8') as file:
            segments = []
            for row in file:
                row_split = [None if cell == "<NA>" else cell for cell in row.split(" ")]
                segment = RttmSegment(
                    type=row_split[0],
                    file=row_split[1],
                    chnl=int(row_split[2]),
                    tbeg=float(row_split[3]),
                    tdur=float(row_split[4]),
                    ortho=row_split[5],
                    stype=row_split[6],
                    name=row_split[7],
                    conf=float(row_split[8]) if row_split[8] is not None else None
                )
                segments.append(segment)

            return cls(segments)


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
        audio_annotation: RttmAnnotation,
        options: Optional[whisper.DecodingOptions] = None,
        show_progress: bool = True
    ) -> List[srt.Subtitle]:
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

        subtitles = []

        for i, seg in tqdm(
            enumerate(audio_annotation.segments),
            total=len(audio_annotation.segments),
            disable=not show_progress
        ):
            # Get start and end frame
            start = int(seg.tbeg * SAMPLE_RATE)
            end = int((seg.tbeg + seg.tdur) * SAMPLE_RATE)

            # Subset audio signal
            audio_sub = audio[start:end]

            output = self.transcriber.transcribe(audio_sub, **asdict(options))

            segment_text = output['text'].strip()

            # Split text into sentences
            sents = re.split(self.sentence_rule, segment_text)

            # Concatenate word timestamps from every segment
            whole_word_timestamps = []

            for segment in output['segments']:
                whole_word_timestamps.extend(segment['whole_word_timestamps'])

            if len(whole_word_timestamps) > 0:
                idx = 0

                for sent in sents:
                    sent_len = len(sent.split(" ")) - 1

                    sent_start = whole_word_timestamps[idx]['timestamp']
                    sent_end = whole_word_timestamps[idx + sent_len]['timestamp']

                    subtitles.append(srt.Subtitle(
                        index=i,
                        start=timedelta(seconds=seg.tbeg + sent_start),
                        end=timedelta(seconds=seg.tbeg + sent_end),
                        content=sent
                    ))

                    idx += sent_len + 1

        return subtitles


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


    @staticmethod
    def write_srt(filename: str, subtitles: List[srt.Subtitle]):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(srt.compose(subtitles))


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

    audio_annotation = RttmAnnotation.from_rttm(args['annotation_path'])

    output = transcriber.apply(
        args['filepath'],
        audio_annotation=audio_annotation,
        options=options,
        show_progress=args['show_progress']
    )


    transcriber.write_srt(
        os.path.join(args['outdir'], os.path.basename(args['filepath']) + '.srt'),
        output
    )


if __name__ == '__main__':
    cli()
