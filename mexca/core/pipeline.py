""" Pipeline class and methods """

import os
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.features import FeaturePitchF0
from mexca.audio.integration import AudioIntegrator
from mexca.audio.speaker_id import SpeakerIdentifier
from mexca.core.exceptions import PipelineError
from mexca.core.output import Multimodal
from mexca.core.preprocessing import Video2AudioConverter
from mexca.text.transcription import AudioTextIntegrator, AudioTranscriber
from mexca.video.extraction import FaceExtractor

class Pipeline:
    def __init__(self, video=None, audio=None, text=None) -> 'Pipeline':
        if text and not audio:
            raise PipelineError('Cannot initialize a "text" component if no "audio" component was specified')
        self.video = video
        self.audio = audio
        self.text = text


    @classmethod
    def from_default(cls, voice='low', language='english'):

        if voice == 'low':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75, pitch_ceiling=300)}
        elif voice == 'high':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=100, pitch_ceiling=500)}
        else:
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75, pitch_ceiling=600)}

        return cls(
            video=FaceExtractor(),
            audio=AudioIntegrator(
                SpeakerIdentifier(),
                VoiceExtractor(features=features)
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(language)
            )
        )


    def apply(self, filepath, remove_audiofile=False) -> 'Multimodal':
        pipeline_result = Multimodal()

        if self.video:
            video_result = self.video.apply(filepath)
            pipeline_result.add(video_result)

        if self.audio:
            with Video2AudioConverter(filepath) as clip:
                audio_path = clip.create_audiofile_path()
                clip.write_audiofile(audio_path)

            if self.video:
                time = video_result['time']
            else:
                time = None

            audio_result = self.audio.apply(audio_path, time)
            pipeline_result.add(audio_result)

            if self.text:
                text_result = self.text.apply(audio_path, audio_result['time'])
                pipeline_result.add(text_result)

            if remove_audiofile:
                os.remove(audio_path)

        return pipeline_result
