"""Extract sentiment from text.
"""

import argparse
import logging
import os
from typing import Optional
import torch
from intervaltree import Interval
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, XLMRobertaForSequenceClassification
from mexca.data import AudioTranscription, SentimentAnnotation, SentimentData
from mexca.utils import ClassInitMessage, str2bool


class SentimentExtractor:
    """Extract sentiment from text.

    Parameters
    ----------
    model_name: str, optional
        The name of the text sequence classification model on Hugging Face hub used for
        sentiment prediction. By default `'cardiffnlp/twitter-xlm-roberta-base-sentiment'`.
    device: torch.device, optional, default=None
        The device on which sentiment extraction is performed. If `None`, defaults to `'cpu'`.

    Attributes
    ----------
    tokenizer: transformers.PreTrainedTokenizer
        The pretrained tokenizer for sequence classification.
        Loaded automatically from `model_name`.

    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[torch.device] = None):
        self.logger = logging.getLogger('mexca.text.extraction.SentimentExtractor')
        if not model_name:
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.logger.debug('Using default pretrained model %s because "model_name=None"', model_name)

        if device is None:
            device = "cpu"

        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Lazy initialization
        self._classifier = None
        self.logger.debug(ClassInitMessage())


    # Initialize pretrained models only when needed
    @property
    def classifier(self) -> XLMRobertaForSequenceClassification:
        """The pretrained sequence classification model for sentiment prediction.
        Loaded automatically from `model_name`.
        """
        if not self._classifier:
            self._classifier = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_8bit=self.device == "cuda"
            )
            self.logger.debug('Initialized sentiment extraction model')

        return self._classifier


    # Delete pretrained models when not needed anymore
    @classifier.deleter
    def classifier(self):
        self._classifier = None
        self.logger.debug('Removed sentiment extraction model')


    def apply(self, transcription: AudioTranscription, show_progress: bool = True) -> SentimentAnnotation:
        """Extract the sentiment from text.

        Iterates over the sentences in the audio transcription and predicts the sentiment
        (negative, neutral, positive).

        Parameters
        ----------
        transcription: AudioTranscription
            The transcription of the speech segments in the audio fie split into sentences.
            Returned by `AudioTranscriber`.
        show_progress: bool, optional, default=True
            Whether a progress bar is displayed or not.

        Returns
        -------
        SentimentAnnotation
            An data class object with the positive, negative, and neutral sentiment scores
            for each sentence.

        """

        sentiment_annotation = SentimentAnnotation()

        for i, sent in tqdm(enumerate(transcription.subtitles), total=len(transcription), disable=not show_progress):
            self.logger.debug('Extracting sentiment for sentence %s', i)
            tokens = self.tokenizer(sent.data.text, return_tensors='pt').to(self.device)
            output = self.classifier(**tokens)
            logits = output.logits.detach().cpu().numpy()
            scores = softmax(logits)[0]
            sentiment_annotation.add(Interval(
                begin=sent.begin,
                end=sent.end,
                data=SentimentData(
                    text=sent.data.text,
                    pos=float(scores[2]),
                    neg=float(scores[0]),
                    neu=float(scores[1])
            )))

        del self.classifier

        return sentiment_annotation


def cli():
    """Command line interface for sentiment extraction.
    See `extract-sentiment -h` for details.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--transcription-path', type=str, required=True, dest='transcription_path')
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('--show-progress', type=str2bool, default=True, dest='show_progress')

    args = parser.parse_args().__dict__

    extractor = SentimentExtractor()

    transcription = AudioTranscription.from_srt(args['transcription_path'])

    output = extractor.apply(transcription, show_progress=args['show_progress'])

    base_name = "_".join(os.path.splitext(os.path.basename(args['transcription_path']))[0].split('_')[:-1])

    output.write_json(os.path.join(args['outdir'], base_name + '_sentiment.json'))


if __name__ == '__main__':
    cli()