"""Extract sentiment from text.
"""

from typing import Optional
from pyannote.core import Annotation
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


    def apply(self, transcription: Annotation) -> Annotation:
        """Extract the sentiment from text.

        Iterates over the segments in the audio annotation and predicts the sentiment
        (negative, neutral, positive) for each segment.

        Parameters
        ----------
        transcription: pyannote.core.Annotation
            The annotation object of the audio file with the added transcription of each segment.
            Returned by `AudioTranscriber`.

        Returns
        -------
        pyannote.core.Annotation
            An annotation object with the positive, negative, and neutral sentiment scores for each
            transribed audio segment.

        """

        for seg in transcription.itersegments():
            for sent in seg.sents:
                tokens = self.tokenizer(sent.text, return_tensors='pt')
                output = self.classifier(**tokens)
                logits = output.logits.detach().numpy()
                scores = softmax(logits)[0]
                sent.sent_pos = scores[2]
                sent.sent_neg = scores[0]
                sent.sent_neu = scores[1]

        return transcription
