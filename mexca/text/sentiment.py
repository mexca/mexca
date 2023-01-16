"""Extract sentiment from text.
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import List, Optional
import srt
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, XLMRobertaForSequenceClassification
from tqdm import tqdm
from mexca.data import AudioTranscription, Sentiment, SentimentAnnotation
from mexca.utils import str2bool


class SentimentExtractor:
    """Extract sentiment from text.

    Parameters
    ----------
    model_name: str, optional
        The name of the text sequence classification model on Hugging Face hub used for
        sentiment prediction. By default `'cardiffnlp/twitter-xlm-roberta-base-sentiment'`.

    Attributes
    ----------
    tokenizer: transformers.PreTrainedTokenizer
        The pretrained tokenizer for sequence classification.
        Loaded automatically from `model_name`.
    classifier: transformers.PreTrainedModel
        The pretrained sequence classification model for sentiment prediction.
        Loaded automatically from `model_name`.


    """
    def __init__(self, model_name: Optional[str] = None):
        if not model_name:
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Lazy initialization
        self._classifier = None


    # Initialize pretrained models only when needed
    @property
    def classifier(self) -> XLMRobertaForSequenceClassification:
        if not self._classifier:
            self._classifier = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return self._classifier


    # Delete pretrained models when not needed anymore
    @classifier.deleter
    def classifier(self):
        self._classifier = None


    def apply(self, transcription: AudioTranscription, show_progress: bool = True) -> SentimentAnnotation:
        """Extract the sentiment from text.

        Iterates over the segments in the audio annotation and predicts the sentiment
        (negative, neutral, positive) for each sentence in each segment.

        Parameters
        ----------
        transcription: pyannote.core.Annotation
            The annotation object of the audio file with the added transcription of each segment.
            Returned by `AudioTranscriber`.
        show_progress: bool, optional, default=True
            Whether a progress bar is displayed or not.

        Returns
        -------
        pyannote.core.Annotation
            An annotation object with the positive, negative, and neutral sentiment scores
            for each sentence in each transribed audio segment.

        """

        sentiment_annotation = SentimentAnnotation()

        for sent in tqdm(transcription.subtitles, total=len(transcription), disable=not show_progress):
            tokens = self.tokenizer(sent.content, return_tensors='pt')
            output = self.classifier(**tokens)
            logits = output.logits.detach().numpy()
            scores = softmax(logits)[0]
            sentiment_annotation.sentiment.append(Sentiment(
                index=sent.index,
                pos=float(scores[2]),
                neg=float(scores[0]),
                neu=float(scores[1])
            ))

        del self.classifier

        return sentiment_annotation


def cli():
    """Command line interface for sentiment extraction.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--transcription-path', type=str, required=True, dest='transcription_path')
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