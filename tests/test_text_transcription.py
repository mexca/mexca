""" Test Audio to text transcription classes and methods """

import os
import subprocess

import numpy as np
import pytest
import whisper

from mexca.data import AudioTranscription, SpeakerAnnotation, TranscriptionData
from mexca.text import AudioTranscriber


class TestAudioTranscription:
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )
    annotation_path = os.path.join(
        "tests",
        "reference_files",
        "test_video_audio_5_seconds_audio_annotation.json",
    )

    annotation = SpeakerAnnotation.from_json(annotation_path)

    @pytest.fixture
    def audio_transcriber(self):
        return AudioTranscriber(whisper_model="tiny")

    def test_lazy_init(self, audio_transcriber):
        assert not audio_transcriber._transcriber
        assert isinstance(audio_transcriber.transcriber, whisper.Whisper)
        del audio_transcriber.transcriber
        assert not audio_transcriber._transcriber

    def test_apply(self, audio_transcriber):
        transcription = audio_transcriber.apply(self.filepath, self.annotation)

        assert isinstance(transcription, AudioTranscription)
        # Only one segment
        for seg in transcription.subtitles.items():
            assert isinstance(seg.data, TranscriptionData)
            assert isinstance(seg.begin, float)
            assert 5.0 >= seg.begin >= 0.0
            assert isinstance(seg.end, float)
            assert 5.0 >= seg.end >= 0.0
            assert isinstance(seg.data.text, str)

    @pytest.fixture
    def timestamps(self):
        return [
            {"start": 0.1, "end": 0.2, "probability": 0.75},
            {"start": 0.3, "end": 0.4, "probability": 0.25},
        ]

    def test_get_timestamp(self, audio_transcriber, timestamps):
        idx = 1
        assert 0.3 == audio_transcriber._get_timestamp(timestamps, idx)
        assert 0.4 == audio_transcriber._get_timestamp(timestamps, idx, "end")

    def test_get_avg_confidence(self, audio_transcriber, timestamps):
        idx = 0
        assert 0.5 == audio_transcriber._get_avg_confidence(
            timestamps, idx, len(timestamps)
        )
        assert np.isnan(
            audio_transcriber._get_avg_confidence(timestamps, idx, 0)
        )

    def test_cli(self):
        out_filename = (
            os.path.splitext(os.path.basename(self.filepath))[0]
            + "_transcription.json"
        )
        subprocess.run(
            [
                "transcribe",
                "-f",
                self.filepath,
                "-a",
                self.annotation_path,
                "-o",
                ".",
            ],
            check=True,
        )
        assert os.path.exists(out_filename)
        os.remove(out_filename)


class TestWhisper:
    model_size = "tiny"
    filepath = os.path.join(
        "tests", "test_files", "test_video_audio_5_seconds.wav"
    )

    @pytest.fixture
    def model(self):
        return whisper.load_model(self.model_size)

    @pytest.fixture
    def stable_model(self):
        return whisper.load_model(self.model_size)

    def test_transcribe(self, model):
        output = model.transcribe(self.filepath, fp16=False)

        # Test entire text of audio and language detection
        assert isinstance(output["text"].strip(), str)
        assert isinstance(output["language"], str)
