""" Test audio sentiment extraction classes and methods """

import pytest
from spacy.language import Vocab
from spacy.tokens import Doc
from mexca.text.sentiment import SequenceClassifier, SentimentExtractor


# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestSequenceClassifier:

    @pytest.fixture
    def classifier(self):
        return SequenceClassifier("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    text = 'Today was a good day!'
    reference = [0.01545323, 0.06419527, 0.9203513]

    def test_properties(self, classifier):
        with pytest.raises(TypeError):
            classifier.model = 3.0

        with pytest.raises(TypeError):
            classifier.tokenizer = 'k'

        with pytest.raises(TypeError):
            classifier.classifier = 'k'


    def test_apply(self, classifier):
        scores = classifier.apply(self.text)
        assert pytest.approx(scores[0].tolist()) == self.reference


@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestSentimentExtractor:

    @pytest.fixture
    def extractor(self):
        return SentimentExtractor()

    doc = Doc(
        Vocab(),
        words=['Today', 'was', 'a', 'good', 'day', '!'],
        spaces=[True, True, True, True, False, False],
        sent_starts=[True, False, False, False, False, False]
    )
    reference = [0.01545323, 0.06419527, 0.9203513]

    def test_properties(self, extractor):
        with pytest.raises(TypeError):
            extractor.roberta = 'k'


    def test_apply(self, extractor):
        sentiment = extractor.apply(self.doc)
        assert pytest.approx(sentiment[0].tolist()) == self.reference
