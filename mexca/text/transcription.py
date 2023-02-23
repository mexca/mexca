"""Transcribe speech from audio to text.
"""

import argparse
import logging
import os
import re
import warnings
from dataclasses import asdict
from typing import Optional, Union
import stable_whisper
import torch
import whisper
from intervaltree import Interval, IntervalTree
from tqdm import tqdm
from whisper.audio import SAMPLE_RATE
from mexca.data import AudioTranscription, SpeakerAnnotation, TranscriptionData
from mexca.utils import ClassInitMessage, optional_str, str2bool


# To filter out shift warnings which do not apply here
warnings.simplefilter('ignore', category=UserWarning)

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

    """
    def __init__(self,
        whisper_model: Optional[str] = 'small',
        device: Optional[Union[str, torch.device]] = 'cpu',
        sentence_rule: Optional[str] = None
    ):
        self.logger = logging.getLogger('mexca.text.transcription.AudioTranscriber')
        self.whisper_model = whisper_model
        self.device = device
        # Lazy initialization
        self._transcriber = None

        if not sentence_rule:
            self.sentence_rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|:)\s"
            self.logger.debug('Using default sentence rule %s because "sentence_rule=None"', self.sentence_rule)
        else:
            self.sentence_rule = sentence_rule

        self.logger.debug(ClassInitMessage())


    # Initialize pretrained models only when needed
    @property
    def transcriber(self) -> whisper.Whisper:
        """The loaded whisper model for audio transcription.
        """
        if not self._transcriber:
            self._transcriber = stable_whisper.load_model(
                self.whisper_model,
                self.device
            )
            self.logger.debug('Initialized %s whisper model for audio transcription', self.whisper_model)

        return self._transcriber


    # Delete pretrained models when not needed anymore
    @transcriber.deleter
    def transcriber(self):
        self._transcriber = None
        self.logger.debug('Removed %s whisper model for audio transcription', self.whisper_model)


    def apply(self, # pylint: disable=too-many-locals
        filepath: str,
        audio_annotation: SpeakerAnnotation,
        language: Optional[str] = None,
        options: Optional[whisper.DecodingOptions] = None,
        show_progress: bool = True
    ) -> AudioTranscription:
        """Transcribe speech in an audio file to text.

        Transcribe each annotated speech segment in the audio file
        and split the transcription into sentences according to `sentence_rule`.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        audio_annotation: SpeakerAnnotation
            The audio annotation object returned the `SpeakerIdentifier` component.
        language: str, optional, default=None
            The language that is transcribed. Ignored if `options.language` is not `None`.
        options: whisper.DecodingOptions, optional
            Options for transcribing the audio file. If `None`, transcription is done without timestamps,
            and with a number format that depends on whether CUDA is available:
            FP16 (half-precision floating points) if available,
            FP32 (single-precision floating points) otherwise.
        show_progress: bool, optional, default=True
            Whether a progress bar is displayed or not.

        Returns
        -------
        AudioTranscription
            A data class object containing transcribed speech segments split into sentences.

        """
        if not options:
            self.logger.debug('Using default options for whisper: No native timestamps and FP16 only if CUDA is available')
            options = self.get_default_options(language=language)

        audio = torch.Tensor(whisper.load_audio(filepath))

        transcription = AudioTranscription(
            filename=filepath,
            subtitles=IntervalTree()
        )

        for i, seg in tqdm(
            enumerate(audio_annotation),
            total=len(audio_annotation),
            disable=not show_progress
        ):
            # Get start and end frame
            start = int(seg.begin * SAMPLE_RATE)
            end = int(seg.end * SAMPLE_RATE)

            # Subset audio signal
            audio_sub = audio[start:end]

            self.logger.debug('Transcribing segment %s from %s to %s', i, seg.begin, seg.end)
            try:
                output = self.transcriber.transcribe(audio_sub, verbose=None, **asdict(options))
            except RuntimeError as exc:
                self.logger.error('Audio waveform too short to be transcribed: %s', exc)
                continue

            self.logger.debug('Detected language: %s', whisper.tokenizer.LANGUAGES[output['language']].title())
            segment_text = output['text'].strip()

            # Split text into sentences
            sents = re.split(self.sentence_rule, segment_text)
            self.logger.debug('Segment text split into %s sentences', len(sents))

            # Concatenate word timestamps from every segment
            whole_word_timestamps = []

            for segment in output['segments']:
                whole_word_timestamps.extend(segment['whole_word_timestamps'])

            if len(whole_word_timestamps) > 0:
                idx = 0

                for j, sent in enumerate(sents):
                    sent_len = len(sent.split(" ")) - 1

                    sent_start = whole_word_timestamps[idx]['timestamp']
                    sent_end = whole_word_timestamps[idx + sent_len]['timestamp']
                    self.logger.debug(
                        'Processing sentence %s from %s to %s with text: %s', j, seg.begin+sent_start, seg.begin+sent_end, sent
                    )

                    if (sent_end - sent_start) > 0:
                        transcription.subtitles.add(Interval(
                            begin=seg.begin + sent_start,
                            end=seg.begin + sent_end,
                            data=TranscriptionData(
                                index=i,
                                text=sent,
                                speaker=seg.data.name
                            )
                        ))
                    else:
                        self.logger.warning('Sentence has duration <= 0 and was not added to transcription')

                    idx += sent_len + 1

        del self._transcriber

        return transcription


    @staticmethod
    def get_default_options(language: Optional[str] = None) -> whisper.DecodingOptions:
        """Set default options for transcription.

        Sets language as well as `without_timestamps=False` and `fp16=torch.cuda.is_available()`.

        Returns
        -------
        whisper.DecodingOptions

        """
        return whisper.DecodingOptions(
            language=language,
            without_timestamps=False,
            fp16=torch.cuda.is_available()
        )


# Adapted from whisper.trascribe.cli
# See: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
def cli():
    """Command line interface for audio transcription.
    See `transcribe -h` for details.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-a', '--annotation-path', type=str, required=True, dest='annotation_path')
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument("--model", default="small", choices=whisper.available_models())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", type=optional_str, default=None)
    parser.add_argument('--sentence-rule', type=optional_str, default=None, dest='sentence_rule')
    parser.add_argument('--show-progress', type=str2bool, default=True, dest='show_progress')

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

    audio_annotation = SpeakerAnnotation.from_rttm(args['annotation_path'])

    output = transcriber.apply(
        args['filepath'],
        audio_annotation=audio_annotation,
        options=options,
        show_progress=args['show_progress']
    )

    output.write_srt(os.path.join(args['outdir'], os.path.splitext(os.path.basename(args['filepath']))[0] + '_transcription.srt'))


if __name__ == '__main__':
    cli()
