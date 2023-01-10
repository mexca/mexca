"""Extract sentiment from text.
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import List, Optional
import srt
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from mexca.data import Sentiment
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
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)


    def apply(self, transcription: List[srt.Subtitle], show_progress: bool = True) -> List[Sentiment]:
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

        sentiment = []

        for sent in tqdm(transcription, total=len(transcription), disable=not show_progress):
            tokens = self.tokenizer(sent.content, return_tensors='pt')
            output = self.classifier(**tokens)
            logits = output.logits.detach().numpy()
            scores = softmax(logits)[0]
            sentiment.append(Sentiment(
                index=sent.index,
                pos=float(scores[2]),
                neg=float(scores[0]),
                neu=float(scores[1])
            ))

        return sentiment


    @staticmethod
    def read_srt(filename: str) -> List[srt.Subtitle]:
        with open(filename, 'r', encoding='utf-8') as file:
            subtitles = srt.parse(file)

            return list(subtitles)


    @staticmethod
    def write_json(filename: str, sentiment: List[Sentiment]):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(json.dumps([asdict(sent) for sent in sentiment]))


def cli():
    """Command line interface for sentiment extraction.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--transcription-path', type=str, required=True, dest='transcription_path')
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('--show-progress', type=str2bool, default=True, dest='show_progress')

    args = parser.parse_args().__dict__

    extractor = SentimentExtractor()

    transcription = extractor.read_srt(args['transcription_path'])

    output = extractor.apply(transcription, show_progress=args['show_progress'])

    extractor.write_json(
        os.path.join(args['outdir'], os.path.basename(args['transcription_path']) + '_sentiment.json'),
        output
    )


if __name__ == '__main__':
    cli()