"""Build a pipeline to extract emotion expression features from a video file.
"""

import logging
import logging.config
import os
from typing import Optional, Tuple, Union
from moviepy.editor import VideoFileClip
from mexca.data import Multimodal
from mexca.utils import ClassInitMessage


class Pipeline:
    """Build a pipeline to extract emotion expression features from a video file.

    Takes either component objects or container component objects (or a mix of both) as input.

    Parameters
    ----------
    face_extractor : FaceExtractor or FaceExtractorContainer, optional, default=None
        Component for detecting and identifying faces as well as extracting facial features.
    speaker_identifier : SpeakerIdentifier or SpeakerIdentifierContainer, optional, default=None
        Component for identifying speech segments and speakers.
    voice_extractor : VoiceExtractor or VoiceExtractorContainer, optional, default=None
        Component for extracting voice features.
    audio_transcriber : AudioTranscriber or AudioTranscriberContainer, optional, default=None
        Component for transcribing speech segments to text.
    sentiment_extractor : SentimentExtractor or SentimentExtractorContainer, optional, default=None
        Component for extracting sentiment from text.

    Examples
    --------
    Create a pipeline with standard components.

    >>> from mexca import Pipeline
    >>> from mexca.audio import SpeakerIdentifier, VoiceExtractor
    >>> from mexca.text import AudioTranscriber, SentimentExtractor
    >>> from mexca.video import FaceExtractor
    >>> num_faces = 2
    >>> num_speaker = 2
    >>> pipeline = Pipeline(
    ...     face_extractor=FaceExtractor(num_faces=num_faces),
    ...     speaker_identifier=SpeakerIdentifier(
    ...         num_speakers=num_speakers
    ...     ),
    ...     voice_extractor=VoiceExtractor(),
    ...     audio_transcriber=AudioTranscriber(),
    ...     sentiment_extractor=SentimentExtractor()
    ... )

    Create a pipeline with container components.

    >>> from mexca import Pipeline
    >>> from mexca.container import AudioTranscriberContainer, FaceExtractorContainer,
    >>>     SentimentExtractorContainer, SpeakerIdentifierContainer, VoiceExtractorContainer
    >>> num_faces = 2
    >>> num_speaker = 2
    >>> pipeline = Pipeline(
    ...     face_extractor=FaceExtractorContainer(num_faces=num_faces),
    ...     speaker_identifier=SpeakerIdentifierContainer(
    ...         num_speakers=num_speakers
    ...     ),
    ...     voice_extractor=VoiceExtractorContainer(),
    ...     audio_transcriber=AudioTranscriberContainer(),
    ...     sentiment_extractor=SentimentExtractorContainer()
    ... )

    Create a pipeline with standard *and* container components.

    >>> from mexca import Pipeline
    >>> from mexca.audio import SpeakerIdentifier, VoiceExtractor
    >>> from mexca.container import AudioTranscriberContainer, FaceExtractorContainer,
    >>>     SentimentExtractorContainer
    >>> num_faces = 2
    >>> num_speaker = 2
    >>> pipeline = Pipeline(
    ...     face_extractor=FaceExtractorContainer(num_faces=num_faces),
    ...     speaker_identifier=SpeakerIdentifier( # standard
    ...         num_speakers=num_speakers
    ...     ),
    ...     voice_extractor=VoiceExtractor(), # standard
    ...     audio_transcriber=AudioTranscriberContainer(),
    ...     sentiment_extractor=SentimentExtractorContainer()
    ... )

    """
    def __init__(self,
        face_extractor: Optional[Union['FaceExtractor', 'FaceExtractorContainer']] = None,
        speaker_identifier: Optional[Union['SpeakerIdentifier', 'SpeakerIdentifierContainer']] = None,
        voice_extractor: Optional[Union['VoiceExtractor', 'VoiceExtractorContainer']] = None,
        audio_transcriber: Optional[Union['AudioTranscriber', 'AudioTranscriberContainer']] = None,
        sentiment_extractor: Optional[Union['SentimentExtractor', 'SentimentExtractorContainer']] = None
    ):
        self.logger = logging.getLogger('mexca.pipeline.Pipeline')
        self.face_extractor = face_extractor
        self.speaker_identifier = speaker_identifier
        self.voice_extractor = voice_extractor
        self.audio_transcriber = audio_transcriber
        self.sentiment_extractor = sentiment_extractor
        self.logger.debug(ClassInitMessage())


    def apply(self, # pylint: disable=too-many-locals
        filepath: str,
        frame_batch_size: int = 1,
        skip_frames: int = 1,
        process_subclip: Tuple[Optional[float]] = (0, None),
        language: Optional[str] = None,
        keep_audiofile: bool = False,
        show_progress: bool = True
    ) -> 'Multimodal':
        """
        Extract emotion expression features from a video file.

        This is the main function to apply the complete mexca pipeline to a video file.

        Parameters
        ----------
        filepath: str
            Path to the video file.
        frame_batch_size: int, default=1
            Size of the batch of video frames that are loaded and processed at the same time.
        skip_frames: int, default=1
            Only process every nth frame, starting at 0.
        process_subclip: tuple, default=(0, None)
            Process only a part of the video clip. Must be the start and end of the subclip in seconds.
            `None` indicates the end of the video.
        language: str, optional, default=None
            The language of the speech that is transcribed.
            If `None`, the language is detected for each speech segment.
        keep_audiofile: bool, default=False
            Keeps the audio file after processing. If False, the audio file is only stored temporarily.
        show_progress: bool, default=True
            Enables progress bars and printing info logging messages to the console.
            The logging is overriden when a custom logger is explicitly created.

        Returns
        -------
        Multimodal
            A data class object that contains the extracted merged features in the `features` attribute.
            See the `Output <https://mexca.readthedocs.io/en/latest/output.html>`_ section for details.

        See Also
        --------
        mexca.data.Multimodal

        Examples
        --------
        >>> import pandas as pd
        >>> filepath = 'path/to/video'
        >>> output = pipeline.apply(filepath)
        >>> assert isinstance(output.features, pd.DataFrame)
        True

        """
        if show_progress:
            logging.getLogger(__name__).setLevel(logging.INFO)
            

        self.logger.info('Starting MEXCA pipeline')
        output = Multimodal(filename=filepath)

        with VideoFileClip(filepath) as clip:
            audio_path = os.path.splitext(filepath)[0] + '.wav'
            subclip = clip.subclip(
                process_subclip[0],
                process_subclip[1]
            )
            self.logger.debug('Reading video file from %s to %s', subclip.start, subclip.end)
            output.duration = subclip.duration
            output.fps = subclip.fps
            output.fps_adjusted = subclip.fps / skip_frames
            time_step = 1 / (subclip.fps / skip_frames)

            if self.speaker_identifier or self.voice_extractor:
                self.logger.debug('Writing audio file')
                subclip.audio.write_audiofile(audio_path, logger=None, fps=16000, ffmpeg_params=["-ac", "1"])
                self.logger.info('Wrote audio file to %s', audio_path)

        if self.face_extractor:
            self.logger.info('Processing video frames')
            video_annotation = self.face_extractor.apply(
                filepath,
                batch_size=frame_batch_size,
                skip_frames=skip_frames,
                process_subclip=process_subclip,
                show_progress=show_progress
            )
            output.video_annotation = video_annotation

        if self.speaker_identifier:
            self.logger.info('Identifying speakers')
            audio_annotation = self.speaker_identifier.apply(audio_path)

            output.audio_annotation = audio_annotation

            if self.audio_transcriber:
                self.logger.info('Transcribing speech segments to text')
                transcription = self.audio_transcriber.apply(
                    audio_path,
                    audio_annotation,
                    language=language,
                    show_progress=show_progress
                )

                output.transcription = transcription

                if self.sentiment_extractor:
                    self.logger.info('Extracting sentiment from transcribed text')
                    sentiment = self.sentiment_extractor.apply(
                        transcription=transcription,
                        show_progress=show_progress
                    )

                    output.sentiment = sentiment

        if self.voice_extractor:
            self.logger.info('Extracting voice features')
            voice_features = self.voice_extractor.apply(
                audio_path,
                time_step=time_step,
                skip_frames=skip_frames
            )

            output.voice_features = voice_features

        output.merge_features()

        if not keep_audiofile and os.path.exists(audio_path):
            self.logger.info('Removing audio file at %s', audio_path)
            os.remove(audio_path)

        self.logger.info('MEXCA pipeline finished')
        return output
