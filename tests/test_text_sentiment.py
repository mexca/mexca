""" Test audio sentiment extraction classes and methods """

import os
import subprocess

import pytest
from intervaltree import Interval, IntervalTree
from transformers import XLMRobertaForSequenceClassification

from mexca.data import (
    AudioTranscription,
    SentimentAnnotation,
    SentimentData,
    TranscriptionData,
)
from mexca.text import SentimentExtractor


# Skip tests on GitHub actions runner for Windows and Linux but
# allow local runs
class TestSentimentExtractor:
    transcription_path = os.path.join(
        "tests",
        "reference_files",
        "test_video_audio_5_seconds_transcription.json",
    )
    reference = {"pos": 0.9203513, "neg": 0.01545322, "neu": 0.06419527}

    @pytest.fixture
    def extractor(self):
        return SentimentExtractor()

    @pytest.fixture
    def transcription(self):
        transcription = AudioTranscription(
            filename=self.transcription_path,
            segments=IntervalTree(
                [
                    Interval(
                        begin=0,
                        end=1,
                        data=TranscriptionData(
                            index=0, text="Today was a good day!", speaker="0"
                        ),
                    )
                ]
            ),
        )

        return transcription

    def test_lazy_init(self, extractor):
        assert not extractor._classifier
        assert isinstance(
            extractor.classifier, XLMRobertaForSequenceClassification
        )
        del extractor.classifier
        assert not extractor._classifier

    def test_apply(self, extractor, transcription):
        sentiment = extractor.apply(transcription)
        # Get first sentence object from first segment
        assert isinstance(sentiment, SentimentAnnotation)

        for sent in sentiment.segments.items():
            assert isinstance(sent.data, SentimentData)
            # Need to use rel=1e-2 due to fluctuation across different runs on platforms
            assert (
                pytest.approx(sent.data.pos, rel=1e-2) == self.reference["pos"]
            )
            assert (
                pytest.approx(sent.data.neg, rel=1e-2) == self.reference["neg"]
            )
            assert (
                pytest.approx(sent.data.neu, rel=1e-2) == self.reference["neu"]
            )

    def test_cli(self):
        out_filename = "test_video_audio_5_seconds_sentiment.json"
        subprocess.run(
            ["extract-sentiment", "-t", self.transcription_path, "-o", "."],
            check=True,
        )
        assert os.path.exists(out_filename)
        os.remove(out_filename)
