""" Pipeline class and methods """

import os
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.features import FeaturePitchF0
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.output import Multimodal
from mexca.core.preprocessing import Video2AudioConverter
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber
from mexca.video.extraction import FaceExtractor


class Pipeline:
    def __init__(self, video=None, audio=None, text=None, time_step=None) -> 'Pipeline':
        self.video = video
        self.audio = audio
        self.text = text

        if not video:
            if audio:
                self.audio.time_step = time_step
            if text:
                self.text.time_step = time_step


    @classmethod
    def from_default(cls, voice='low', language='english'):

        if voice == 'low':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75, pitch_ceiling=300)}
        elif voice == 'high':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=100, pitch_ceiling=500)}
        else:
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75, pitch_ceiling=600)}

        return cls(
            video=FaceExtractor(min_clusters=1),
            audio=AudioIntegrator(
                SpeakerIdentifier(),
                VoiceExtractor(features=features)
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(language)
            )
        )


    def apply( # pylint: disable=too-many-arguments
            self,
            filepath,
            skip_frames=1,
            process_subclip=(0, None),
            keep_audiofile=False,
            show_video_progress=True,
            show_audio_progress=True
        ) -> 'Multimodal':
        """
        Runs the video, audio and text pipelines

        Parameters
        ---------------------------------
        keep_audiofile: bool
            Keeps in memory the audio file after processing. Default to False.
        skip_video_frames: float
            Tells video detector to process only every nth frame. Default to 1.
        verbose: bool,
            Enables a progress bar. Default to False.
        """
        pipeline_result = Multimodal()

        if self.video:
            print('Analyzing video ...')
            video_result = self.video.apply(
                filepath,
                skip_frames=skip_frames,
                process_subclip=process_subclip,
                show_progress=show_video_progress
            )
            pipeline_result.add(video_result)
            print('Video done')
            time = video_result['time']
        else:
            time = None

        if self.audio or self.text:
            with Video2AudioConverter(filepath) as clip:
                audio_path = clip.create_audiofile_path()
                # Use subclip if `process_subclip` is provided (default uses entire clip)
                clip.subclip(process_subclip[0], process_subclip[1]).write_audiofile(audio_path)

            if self.audio:
                print('Analyzing audio ...')
                audio_result = self.audio.apply(audio_path, time, show_audio_progress)
                pipeline_result.add(audio_result)
                print('Audio done')

            if self.text:
                print('Analyzing text ...')
                text_result = self.text.apply(audio_path, time)
                pipeline_result.add(text_result)
                print('Text done')

            if not keep_audiofile:
                os.remove(audio_path)

            # Match face ids with speaker ids -> id vector
            pipeline_result.match_faces_speakers()

        return pipeline_result
