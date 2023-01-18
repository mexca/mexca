""" Test Audio to text transcription classes and methods """

import os
import subprocess
from datetime import timedelta
import pytest
import srt
import stable_whisper
import whisper
from mexca.data import AudioTranscription, SpeakerAnnotation, TranscriptionData
from mexca.text import AudioTranscriber


class TestAudioTranscription:
    filepath = os.path.join(
        'tests', 'test_files', 'test_video_audio_5_seconds.wav'
    )
    annotation_path = os.path.join(
        'tests', 'reference_files', 'test_video_audio_5_seconds_audio_annotation.rttm'
    )
    
    annotation = SpeakerAnnotation.from_rttm(annotation_path)

    
    @pytest.fixture
    def audio_transcriber(self):
        return AudioTranscriber(whisper_model='tiny')


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


    def test_cli(self):
        out_filename = os.path.splitext(os.path.basename(self.filepath))[0] + '_transcription.srt'
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
