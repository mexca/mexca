"""Extract sentiment from text.
"""

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast


class SequenceClassifier:
    """A classifier for text sequences.

    Wrapper class for ``transformers.AutoTokenizer`` and
    ``AutoModelForSequenceClassification``.

    Parameters
    ----------
    model: str
        The name of the sequence classification model (e.g., on HuggingFace Hub).

    """
    def __init__(self, model) -> 'SequenceClassifier':
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(self.model)


    @property
    def model(self):
        return self._model


    @model.setter
    def model(self, new_model):
        if isinstance(new_model, str):
            self._model = new_model
        else:
            raise TypeError('Can only set "model" to str')


    @property
    def tokenizer(self):
        return self._tokenizer


    @tokenizer.setter
    def tokenizer(self, new_tokenizer):
        if isinstance(new_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self._tokenizer = new_tokenizer
        else:
            raise TypeError("""Can only set "tokenizer" to instance of "PreTrainedTokenizer"
                or "PreTrainedTokenizerFast" class""")


    @property
    def classifier(self):
        return self._classifier


    @classifier.setter
    def classifier(self, new_classifier):
        if isinstance(new_classifier, PreTrainedModel):
            self._classifier = new_classifier
        else:
            raise TypeError("""Can only set "classifier" to instance of
                "PreTrainedModel" class""")


    def apply(self, text):
        """Classify text sequences.

        Parameters
        ----------
        text: str or list[str]
            Text strings to be classified.

        Returns
        -------
        np.ndarray
            An array of probabilities for each class of the classifier.

        """
        tokens = self.tokenizer(text, return_tensors='pt')
        output = self.classifier(**tokens)
        logits = output.logits.detach().numpy()
        scores = softmax(logits)

        return scores


class SentimentExtractor:
    """A sentiment extractor for text.
    """
    def __init__(self) -> 'SentimentExtractor':
        self.roberta = SequenceClassifier(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )


    @property
    def roberta(self):
        return self._roberta


    @roberta.setter
    def roberta(self, new_roberta):
        if isinstance(new_roberta, SequenceClassifier):
            self._roberta = new_roberta
        else:
            raise TypeError('Can only set "roberta" to instance of "SequenceClassifier" class')


    def apply(self, docs):
        """Extract the sentiment from text.

        Iterates over the sentences in the text and predicts the sentiment
        (negative, neutral, positive) for each sentence.

        Parameters
        ----------
        docs: spacy.tokens.Doc
            A ``Doc`` instance with a `sents` attribute.

        Returns
        -------
        np.ndarray
            Array containing sentiment probabilities with shape N x 3 where N is the number of sentences in the text.
            The order of sentiments is negative, neutral, positive.

        """
        sentiment = []

        for sent in docs.sents:
            scores = self.roberta.apply(sent.text)[0]
            sentiment.append(scores)

        return np.array(sentiment)
