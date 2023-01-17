"""Build a pipeline to extract emotion expression features from a video file.
"""

import logging
import logging.config
import os
from typing import Optional, Tuple
from moviepy.editor import VideoFileClip
from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.data import Multimodal
from mexca.text import AudioTranscriber, SentimentExtractor
from mexca.utils import ClassInitMessage
from mexca.video import FaceExtractor


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
    def __init__(self,
        face_extractor: FaceExtractor = None,
        speaker_identifier: SpeakerIdentifier = None,
        voice_extractor: VoiceExtractor = None,
        audio_transcriber: AudioTranscriber = None,
        sentiment_extractor: SentimentExtractor = None
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
            output.fps_adjusted = int(subclip.fps / skip_frames)
            time_step = 1/int(subclip.fps / skip_frames)

            if self.speaker_identifier or self.voice_extractor:
                self.logger.debug('Writing audio file')
                subclip.audio.write_audiofile(audio_path, logger=None)
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
                    audio_annotation=audio_annotation,
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
                time_step=time_step
            )

            output.voice_features = voice_features

        output.merge_features()

        if not keep_audiofile and os.path.exists(audio_path):
            self.logger.info('Removing audio file at %s', audio_path)
            os.remove(audio_path)

        self.logger.info('MEXCA pipeline finished')
        return output
