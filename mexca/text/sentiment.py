"""Extract sentiment from text.
"""

import argparse
import os
from typing import Optional
import pyannote.core.json
from pyannote.core import Annotation
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


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


    def apply(self, transcription: Annotation, show_progress: bool = True) -> Annotation:
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

        for seg in tqdm(transcription.itersegments(), total=len(transcription), disable=not show_progress):
            for sent in seg.sents:
                tokens = self.tokenizer(sent.text, return_tensors='pt')
                output = self.classifier(**tokens)
                logits = output.logits.detach().numpy()
                scores = softmax(logits)[0]
                sent.sent_pos = scores[2]
                sent.sent_neg = scores[0]
                sent.sent_neu = scores[1]

        return transcription


def cli():
    """Command line interface for sentiment extraction.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--annotation-path', type=str, required=True, dest='annotation_path')
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('--show-progress', type=str, default=None, dest='show_progress')

    args = parser.parse_args().__dict__

    extractor = SentimentExtractor()

    audio_annotation = pyannote.core.json.load_from(args['annotation_path'])

    output = extractor.apply(audio_annotation, show_progress=args['show_progress'])

    pyannote.core.json.dump_to(
        output,
        os.path.join(args['outdir'], os.path.basename(args['annotation_path']) + '.json')
    )


if __name__ == '__main__':
    cli()