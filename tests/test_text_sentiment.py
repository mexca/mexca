""" Test audio sentiment extraction classes and methods """

import os
import subprocess
import pytest
from intervaltree import Interval, IntervalTree
from transformers import XLMRobertaForSequenceClassification
from mexca.data import AudioTranscription, SentimentData, SentimentAnnotation, TranscriptionData
from mexca.text import SentimentExtractor


# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestSentimentExtractor:
    transcription_path = os.path.join(
        'tests', 'reference_files', 'test_video_audio_5_seconds_transcription.srt'
    )
    reference = {
        'pos': 0.9203513,
        'neg': 0.01545323,
        'neu': 0.06419527
    }


    @pytest.fixture
    def extractor(self):
        return SentimentExtractor()


    @pytest.fixture
    def transcription(self):
        transcription = AudioTranscription(
            filename=self.transcription_path,
            subtitles=IntervalTree([
                Interval(
                    begin=0,
                    end=1,
                    data=TranscriptionData(
                        index=0,
                        text='Today was a good day!',
                        speaker='0'
                    )
                )
            ])
        )

        return transcription


    def test_lazy_init(self, extractor):
        assert not extractor._classifier
        assert isinstance(extractor.classifier, XLMRobertaForSequenceClassification)
        del extractor.classifier
        assert not extractor._classifier


    def test_apply(self, extractor, transcription):
        sentiment = extractor.apply(transcription)
        # Get first sentence object from first segment
        assert isinstance(sentiment, SentimentAnnotation)
        
        for sent in sentiment.items():
            assert isinstance(sent.data, SentimentData)
            assert pytest.approx(sent.data.pos) == self.reference['pos']
            assert pytest.approx(sent.data.neg) == self.reference['neg']
            assert pytest.approx(sent.data.neu) == self.reference['neu']


    def test_cli(self):
        out_filename = 'test_video_audio_5_seconds_sentiment.json'
        subprocess.run(['extract-sentiment', '-t', self.transcription_path, '-o', '.'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
