""" Test audio sentiment extraction classes and methods """

import pytest
from pyannote.core import Annotation
from mexca.text.sentiment import SentimentExtractor
from mexca.text.transcription import Sentence, TranscribedSegment


# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestSentimentExtractor:

    @pytest.fixture
    def extractor(self):
        return SentimentExtractor()

    @pytest.fixture
    def annotation(self):
        seg = TranscribedSegment(
            start=0.0,
            end=1.0,
            text='Today was a good day!',
            lang='en',
            sents=[Sentence(text='Today was a good day!', start=0.0, end=1.0)]
        )
        ann = Annotation()
        ann[seg, 'trackA'] = 'speaker1'

        return(ann)

    reference = {
        'pos': 0.9203513,
        'neg': 0.01545323,
        'neu': 0.06419527
    }


    def test_apply(self, extractor, annotation):
        sentiment_annotation = extractor.apply(annotation)
        # Get first sentence object from first segment
        sentence = list(sentiment_annotation.itersegments())[0].sents[0]
        assert pytest.approx(sentence.sent_pos) == self.reference['pos']
        assert pytest.approx(sentence.sent_neg) == self.reference['neg']
        assert pytest.approx(sentence.sent_neu) == self.reference['neu']
