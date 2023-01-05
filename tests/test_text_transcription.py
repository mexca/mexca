""" Test Audio to text transcription classes and methods """

import os
import subprocess
from datetime import timedelta
import pytest
import srt
import stable_whisper
import whisper
from mexca.text.transcription import AudioTranscriber, RttmAnnotation


class TestAudioTranscription:
    audio_transcriber = AudioTranscriber(whisper_model='tiny')
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )
    annotation_path = os.path.join(
        'tests', 'reference_files', 'annotation_video_audio_5_seconds.rttm'
    )
    
    annotation = RttmAnnotation.from_rttm(annotation_path)

    def test_apply(self):
        transcription = self.audio_transcriber.apply(self.filepath, self.annotation)

        assert isinstance(transcription, list)
        # Only one segment
        assert isinstance(transcription[0], srt.Subtitle)
        assert isinstance(transcription[0].start, timedelta)
        assert isinstance(transcription[0].end, timedelta)
        assert isinstance(transcription[0].content, str)


    def test_cli(self):
        out_filename = os.path.basename(self.filepath) + '.srt'
        subprocess.run(['transcribe', '-f', self.filepath,
                        '-a', self.annotation_path, '-o', '.'], check=True)
        assert os.path.exists(out_filename)
        os.remove(out_filename)


class TestWhisper:
    model_size = 'tiny'
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )


    @pytest.fixture
    def model(self):
        return whisper.load_model(self.model_size)


    @pytest.fixture
    def stable_model(self):
        return stable_whisper.load_model(self.model_size)


    def test_transcribe(self, model):
        output = model.transcribe(self.filepath, fp16=False)

        # Test entire text of audio and language detection
        assert isinstance(output['text'].strip(), str)
        assert isinstance(output['language'], str)


    def test_word_ts(self, stable_model):
        output = stable_model.transcribe(self.filepath, fp16=False)
        # Test word level timestamps of first segment
        first_segment = output['segments'][0]

        assert 'whole_word_timestamps' in first_segment

        first_segment_ts = output['segments'][0]['whole_word_timestamps']
        # Test first token of first segment
        assert isinstance(first_segment_ts[0]['word'], str)
        assert isinstance(first_segment_ts[0]['timestamp'], float)
