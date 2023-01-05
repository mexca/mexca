""" Test audio sentiment extraction classes and methods """

import os
import srt
import subprocess
from datetime import timedelta
import pytest
from mexca.text import SentimentExtractor


# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
@pytest.mark.skip_env('runner')
@pytest.mark.skip_os(['Windows', 'Linux'])
class TestSentimentExtractor:
    @pytest.fixture
    def extractor(self):
        return SentimentExtractor()


    @pytest.fixture
    def transcription(self):
        sent = srt.Subtitle(
            index=0,
            start=timedelta(seconds=0),
            end=timedelta(seconds=1),
            content='Today was a good day!'
        )

        return [sent]

    reference = {
        'pos': 0.9203513,
        'neg': 0.01545323,
        'neu': 0.06419527
    }


    def test_apply(self, extractor, transcription):
        sentiment = extractor.apply(transcription)
        # Get first sentence object from first segment
        sentence = sentiment[0]
        assert pytest.approx(sentence.pos) == self.reference['pos']
        assert pytest.approx(sentence.neg) == self.reference['neg']
        assert pytest.approx(sentence.neu) == self.reference['neu']


    def test_cli(self):
        transcription_path = os.path.join(
            'tests', 'reference_files', 'transcription_video_audio_5_seconds.srt'
        )
        out_filename = os.path.basename(transcription_path) + '_sentiment.json'
        subprocess.run(['extract-sentiment', '-a', transcription_path, '-o', '.'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
