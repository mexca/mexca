"""Build a pipeline to extract emotion expression features from a video file.
"""

import os
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.features import FeaturePitchF0
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.output import Multimodal
from mexca.core.preprocessing import Video2AudioConverter
from mexca.text.sentiment import SentimentExtractor
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber
from mexca.video.extraction import FaceExtractor


class Pipeline:
    """Build a pipeline to extract emotion expression features from a video file.

    Parameters
    ----------
    video: optional, default=None
        A class instance that processes the frames of the video file.
    audio: optional, default=None
        A class instance that process the audio signal of the video file.
    text: optional, default=None
        A class instance that transcribes the speech in the audio signal
        and matches the transcription to frames.

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
    ...         audio_transcriber=AudioTranscriber(),
    ...         sentiment_extractor=SentimentExtractor()
    ...     )
    ... )

    """
    def __init__(self, video=None, audio=None, text=None) -> 'Pipeline':
        self.video = video
        self.audio = audio
        self.text = text


    @property
    def video(self):
        return self._video


    @video.setter
    def video(self, new_video):
        if isinstance(new_video, FaceExtractor) or new_video is None:
            self._video = new_video
        else:
            raise ValueError('Can only set "video" to an instance of "FaceExtractor" class or None')


    @property
    def audio(self):
        return self._audio


    @audio.setter
    def audio(self, new_audio):
        if isinstance(new_audio, AudioIntegrator) or new_audio is None:
            self._audio = new_audio
        else:
            raise ValueError('Can only set "audio" to an instance of "AudioIntegrator" class or None')


    @property
    def text(self):
        return self._text


    @text.setter
    def text(self, new_text):
        if isinstance(new_text, AudioTextIntegrator) or new_text is None:
            self._text = new_text
        else:
            raise ValueError('Can only set "text" to an instance of "AudioTextIntegrator" class or None')


    @classmethod
    def from_default(cls, voice='low', use_auth_token=True):
        """Constructor method to create a pipeline with default components and settings.

        This method is a convenience wrapper for creating a standard pipeline.

        Parameters
        ----------
        voice: {'low', 'high'} or str, default='low'
            The expected frequency spectrum of the voices in the video.
        use_auth_token: bool or str, default=True
            Whether to use the HuggingFace authentication token stored on the machine (if bool) or
            a HuggingFace authentication token with access to the models ``pyannote/speaker-diarization``
            and ``pyannote/segmentation`` (if str).

        Returns
        -------
        A ``Pipeline`` class instance.

        Notes
        -----
        This constructor method requires pretrained models for speaker diarization and segmentation from HuggingFace.
        To download the models accept the user conditions on `<hf.co/pyannote/speaker-diarization>`_ and
        `<hf.co/pyannote/segmentation>`_. Then generate an authentication token on `<hf.co/settings/tokens>`_.

        Examples
        --------
        >>> from mexca.core.pipeline import Pipeline
        >>> pipeline = Pipeline().from_default()

        """

        if voice == 'low':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75.0, pitch_ceiling=300.0)}
        elif voice == 'high':
            features = {'pitchF0': FeaturePitchF0(pitch_floor=100.0, pitch_ceiling=500.0)}
        else:
            features = {'pitchF0': FeaturePitchF0(pitch_floor=75.0, pitch_ceiling=600.0)}

        return cls(
            video=FaceExtractor(min_clusters=1),
            audio=AudioIntegrator(
                SpeakerIdentifier(num_speakers=2, use_auth_token=use_auth_token),
                VoiceExtractor(features=features)
            ),
            text=AudioTextIntegrator(
                audio_transcriber=AudioTranscriber(),
                sentiment_extractor=SentimentExtractor()
            )
        )


    def apply( # pylint: disable=too-many-arguments, disable=too-many-locals
            self,
            filepath,
            skip_frames=1,
            process_subclip=(0, None),
            keep_audiofile=False,
            show_video_progress=True,
            show_audio_progress=True,
            show_text_progress=True
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
        See the `Output <https://mexca.readthedocs.io/en/latest/output.html>`_ section for details.

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

        if self.audio:
            with Video2AudioConverter(filepath) as clip:
                audio_path = clip.create_audiofile_path()
                # Use subclip if `process_subclip` is provided (default uses entire clip)
                clip.subclip(process_subclip[0], process_subclip[1]).write_audiofile(audio_path)

            if self.audio:
                print('Analyzing audio ...')
                audio_annotation, audio_result = self.audio.apply(audio_path, time, show_audio_progress)
                pipeline_result.add(audio_result)
                print('Audio done')

                if self.text:
                    print('Analyzing text ...')
                    text_result = self.text.apply(audio_path, audio_annotation, time, None, show_text_progress)
                    pipeline_result.add(text_result)
                    print('Text done')

            if not keep_audiofile:
                os.remove(audio_path)

        # Match face ids with speaker ids -> id vector
        if self.video and self.audio:
            pipeline_result.match_faces_speakers()

        return pipeline_result
