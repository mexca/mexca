"""Transcribe speech from audio to text.
"""

import argparse
import logging
import os
import re
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import faster_whisper
import numpy as np
import torch
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.transcribe import TranscriptionOptions, Word
from intervaltree import Interval, IntervalTree
from tqdm import tqdm

from mexca.data import AudioTranscription, SpeakerAnnotation, TranscriptionData
from mexca.utils import ClassInitMessage, optional_str, str2bool

SAMPLE_RATE = 16_000


# To filter out shift warnings which do not apply here
warnings.simplefilter("ignore", category=UserWarning)


class AudioTranscriber:
    """Transcribe speech from audio to text.

    Parameters
    ----------
    whisper_model: str, optional, default='small'
        The name of the whisper model that is used for transcription. Available models are
        `['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']`.
    device: str or torch.device, default='cpu'
        The name of the device onto which the whisper model should be loaded and run. If CUDA support is
        available, this can be `'cuda'`, otherwise use `'cpu'` (the default).
    compute_type: str, default='default'
        The quantization type used for computation. Infers the type from the pretrained model for `'default'`. See this
        `link <https://opennmt.net/CTranslate2/quantization.html>`_ for details.
    sentence_rule: str, optional
        A regular expression used to split segment transcripts into sentences. If `None` (default), it splits
        the text at all '.', '?', '!', and ':' characters that are followed by whitespace characters. It
        omits single or multiple words abbreviated with dots (e.g., 'Nr. ' and 'e.g. ').

    """

    def __init__(
        self,
        whisper_model: Optional[str] = "small",
        device: Union[str, torch.device] = "cpu",
        compute_type: str = "default",
        sentence_rule: Optional[str] = None,
    ):
        self.logger = logging.getLogger(
            "mexca.text.transcription.AudioTranscriber"
        )
        self.whisper_model = whisper_model
        self.device = device
        self.compute_type = compute_type
        # Lazy initialization
        self._transcriber = None

        if not sentence_rule:
            self.sentence_rule = (
                r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|:)\s"
            )
            self.logger.debug(
                'Using default sentence rule %s because "sentence_rule=None"',
                self.sentence_rule,
            )
        else:
            self.sentence_rule = sentence_rule

        self.logger.debug(ClassInitMessage())

    # Initialize pretrained models only when needed
    @property
    def transcriber(self) -> WhisperModel:
        """The loaded whisper model for audio transcription."""
        if not self._transcriber:
            self._transcriber = WhisperModel(
                self.whisper_model,
                device=self.device.type
                if isinstance(self.device, torch.device)
                else str(self.device),  # accepts only string, not torch.device
                compute_type=self.compute_type,
            )
            self.logger.debug(
                "Initialized %s whisper model for audio transcription",
                self.whisper_model,
            )

        return self._transcriber

    # Delete pretrained models when not needed anymore
    @transcriber.deleter
    def transcriber(self):
        self._transcriber = None
        self.logger.debug(
            "Removed %s whisper model for audio transcription",
            self.whisper_model,
        )

    # pylint: disable=too-many-locals
    def apply(
        self,
        filepath: str,
        audio_annotation: SpeakerAnnotation,
        language: Optional[str] = None,
        options: Optional[TranscriptionOptions] = None,
        show_progress: bool = True,
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
            options = {}
        else:
            options = asdict(options)

        audio = decode_audio(filepath, sampling_rate=SAMPLE_RATE)

        transcription = AudioTranscription(
            filename=filepath, segments=IntervalTree()
        )

        for i, seg in tqdm(
            enumerate(audio_annotation.segments),
            total=len(audio_annotation.segments),
            disable=not show_progress,
        ):
            # Get segment length
            seg_length = seg.end - seg.begin

            # Get start and end frame
            start = int(seg.begin * SAMPLE_RATE)
            end = int(seg.end * SAMPLE_RATE)

            # Subset audio signal
            audio_sub = audio[start:end]

            self.logger.debug(
                "Transcribing segment %s from %s to %s", i, seg.begin, seg.end
            )
            try:
                output, info = self.transcriber.transcribe(
                    audio_sub,
                    language=language,
                    word_timestamps=True,
                    # verbose=None,
                    **options,
                )
            except RuntimeError as exc:
                if seg_length < 0.02:
                    self.logger.error(
                        "Audio waveform too short to be transcribed: %s", exc
                    )
                else:
                    self.logger.error(
                        "The operator aten::_index_put_impl_ is not current implemented for the MPS device"
                    )
                continue

            self.logger.debug(
                "Detected language: %s",
                info.language,
            )

            for out_seg in output:
                text = out_seg.text.strip()

                # Split text into sentences
                sents = re.split(self.sentence_rule, text)
                # self.logger.debug(
                #     "Segment text split into %s sentences", len(sents)
                # )

                # Concatenate word timestamps from every segment
                whole_word_timestamps = []

                for word in out_seg.words:
                    whole_word_timestamps.append(word)

                if len(whole_word_timestamps) > 0:
                    idx = 0

                    for j, sent in enumerate(sents):
                        sent_len = len(sent.split(" ")) - 1

                        # Get timestamp of first word in sentence (BEFORE the first word is spoken - 'start')
                        sent_start = self._get_timestamp(
                            whole_word_timestamps, idx, timestamp_type="start"
                        )
                        # Get timestamp of last word in sentence (AFTER the last word is spoken - 'end')
                        sent_end = self._get_timestamp(
                            whole_word_timestamps,
                            (idx + sent_len),
                            timestamp_type="end",
                        )

                        self.logger.debug(
                            "Processing sentence %s from %s to %s with text: %s",
                            j,
                            seg.begin + sent_start,
                            seg.begin + sent_end,
                            sent,
                        )

                        if (sent_end - sent_start) > 0:
                            # Calculate average probability of transcription accuracy for sentence
                            conf = self._get_avg_confidence(
                                whole_word_timestamps, idx, sent_len
                            )
                            # Add transcription to output
                            transcription.segments.add(
                                Interval(
                                    begin=seg.begin + sent_start,
                                    end=seg.begin + sent_end,
                                    data=TranscriptionData(
                                        index=i,
                                        text=sent,
                                        speaker=seg.data.name,
                                        confidence=conf,
                                    ),
                                )
                            )
                        else:
                            self.logger.warning(
                                "Sentence has duration <= 0 and was not added to transcription"
                            )

                        idx += sent_len + 1

        del self.transcriber

        return transcription

    @staticmethod
    def _get_timestamp(
        word_timestamps: List[Dict[str, Union[str, float]]],
        idx: int,
        timestamp_type: str = "start",
    ) -> float:
        # get word-level timestamp for the word located at index idx in sequence list of words
        return getattr(word_timestamps[idx], timestamp_type)

    @staticmethod
    def _get_avg_confidence(
        word_timestamps: List[Word],
        idx: int,
        sentence_len: int,
    ) -> float:
        # Computes the average probability / accuracy of
        # transcription for a given sentence. Sums the
        # probabilities for individual words in the sentence
        # and divide by the sentence length
        if len(word_timestamps) == 0 or sentence_len == 0:
            return np.nan

        total = 0.0
        for j, word in enumerate(word_timestamps):
            if idx <= j < sentence_len:
                total += word.probability

        return total / sentence_len


# Adapted from whisper.trascribe.cli
# See: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
def cli():
    """Command line interface for audio transcription.
    See `transcribe -h` for details.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-f", "--filepath", type=str, required=True)
    parser.add_argument(
        "-a",
        "--annotation-path",
        type=str,
        required=True,
        dest="annotation_path",
    )
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument(
        "--model",
        default="small",
        choices=faster_whisper.utils.available_models(),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", type=optional_str, default=None)
    parser.add_argument(
        "--sentence-rule", type=optional_str, default=None, dest="sentence_rule"
    )
    parser.add_argument(
        "--show-progress", type=str2bool, default=True, dest="show_progress"
    )

    args = parser.parse_args().__dict__

    transcriber = AudioTranscriber(
        whisper_model=args["model"],
        device=args["device"],
        sentence_rule=args["sentence_rule"],
    )

    audio_annotation = SpeakerAnnotation.from_json(
        args["annotation_path"], extra_filename=args["filepath"]
    )

    output = transcriber.apply(
        args["filepath"],
        audio_annotation=audio_annotation,
        language=args["language"],
        show_progress=args["show_progress"],
    )

    output.write_json(
        os.path.join(
            args["outdir"],
            os.path.splitext(os.path.basename(args["filepath"]))[0]
            + f"_{output.serialization_name()}.json",
        )
    )


if __name__ == "__main__":
    cli()
