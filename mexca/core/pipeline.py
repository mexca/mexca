"""Build a pipeline to extract emotion expression features from a video file.
"""

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
    """Build a pipeline to extract emotion expression features from a video file.
    """
    def __init__(self, video=None, audio=None, text=None) -> 'Pipeline':
        """Create a class instance containing the pipeline.

        Parameters
        ----------
        video: optional, default=None
            A class instance that processes the frames of the video file.
        audio: optional, default=None
            A class instance that process the audio signal of the video file.
        text: optional, default=None
            A class instance that transcribes the speech in the audio signal
            and matches the transcription to frames.

        Returns
        -------
        A ``Pipeline`` class instance.

        Examples
        --------
        >>> from mexca.audio.extraction import VoiceExtractor
        >>> from mexca.audio.identification import SpeakerIdentifier
        >>> from mexca.audio.integration import AudioIntegrator
        >>> from mexca.core.pipeline import Pipeline
        >>> from mexca.text.transcription import AudioTextIntegrator
        >>> from mexca.text.transcription import AudioTranscriber
        >>> from mexca.video.extraction import FaceExtractor
        >>> pipeline = Pipeline(
        ...     video=FaceExtractor(),
        ...     audio=AudioIntegrator(
        ...         SpeakerIdentifier(),
        ...         VoiceExtractor()
        ...     ),
        ...     text=AudioTextIntegrator(
        ...         audio_transcriber=AudioTranscriber('english')
        ...     )
        ... )

        """
        self.video = video
        self.audio = audio
        self.text = text

    @classmethod
    def from_default(cls, voice='low', language='english'):
        """Constructor method to create a pipeline with default components and settings.

        This method is a convenience wrapper for creating a standard pipeline.

        Parameters
        ----------
        voice: {'low', 'high'} or str, default='low'
            The expected frequency spectrum of the voices in the video.
        language: {'english', 'dutch'}
            The language of the speech in the video. Currently available are English and Dutch.

        Returns
        -------
        A ``Pipeline`` class instance.

        Examples
        --------
        >>> from mexca.core.pipeline import Pipeline
        >>> pipeline = Pipeline().from_default(language='english')

        """

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
        Extract emotion expression features from a video file.

        This is the main function to apply the complete mexca pipeline to a video file.

        Parameters
        ----------
        filepath: str or path
            Path to the video file.
        skip_frames: int, default=1
            Forces the video component to only process every nth frame.
        process_subclip: tuple, default=(0, None)
            Process only a part of the video clip.
            See `moviepy.editor.VideoFileClip
            <https://moviepy.readthedocs.io/en/latest/ref/VideoClip/VideoClip.html#videofileclip>`_ for details.
        keep_audiofile: bool, default=False
            Keeps the audio file after processing. If False, the audio file is only temporary.
        show_video_progress: bool, default=True
            Enables a progress bar for video processing.
        show_audio_progress: bool, default=True
            Enables a progress bar for audio processing.

        Returns
        -------
        A ``Multimodal`` class instance that contains the extracted features in the `features` attribute.

        See Also
        --------
        mexca.video.extraction: Extract facial expression features.
        mexca.audio.extraction: Extract voice features.

        Examples
        --------
        >>> filepath = 'path/to/video'
        >>> output = pipeline.apply(filepath)
        >>> output.features
        {'frame': [0, 1, 2, ...], 'time': [0.04, 0.08, 0.12, ...], ...} # Dictionary with extracted features

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
        if self.video and self.audio:
            pipeline_result.match_faces_speakers()

        return pipeline_result
