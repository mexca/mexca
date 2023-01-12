"""Build a pipeline to extract emotion expression features from a video file.
"""

import os
from dataclasses import asdict
from functools import reduce   
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.data import Multimodal
from mexca.text import AudioTranscriber, SentimentExtractor
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
        self.face_extractor = face_extractor
        self.speaker_identifier = speaker_identifier
        self.voice_extractor = voice_extractor
        self.audio_transcriber = audio_transcriber
        self.sentiment_extractor = sentiment_extractor


    def merge_features(self, output: Multimodal):
        dfs = []

        if output.video_annotation:
            dfs.append(pd.DataFrame(asdict(output.video_annotation)).set_index('frame'))

        if output.audio_annotation:
            audio_annotation_dict = {
                "frame": [],
                "segment_start": [],
                "segment_end": [],
                "segment_speaker_label": []
            }

            time = np.arange(0.0, output.duration, 1/output.fps_adjusted, dtype=np.float32)
            frame = np.arange(0, output.duration*output.fps, output.fps_adjusted, dtype=np.int32)

            if output.transcription:
                text_features_dict = {
                    "frame": [],
                    "span_start": [],
                    "span_end": [],
                    "span_text": []
                }

                if output.sentiment:
                    text_features_dict['span_sent_pos'] = []
                    text_features_dict['span_sent_neg'] = []
                    text_features_dict['span_sent_neu'] = []

            for i, t in zip(frame, time):
                for seg in output.audio_annotation.segments:
                    seg_end = seg.tbeg + seg.tdur
                    if seg.tbeg <= t <= seg_end:
                        audio_annotation_dict['frame'].append(i)
                        audio_annotation_dict['segment_start'].append(seg.tbeg)
                        audio_annotation_dict['segment_end'].append(seg_end)
                        audio_annotation_dict['segment_speaker_label'].append(seg.name)

                if output.transcription and output.sentiment:
                    for span, sent in zip(output.transcription.subtitles, output.sentiment.sentiment):
                        if span.start.total_seconds() <= t <= span.end.total_seconds():
                            text_features_dict['frame'].append(i)
                            text_features_dict['span_start'].append(span.start.total_seconds())
                            text_features_dict['span_end'].append(span.end.total_seconds())
                            text_features_dict['span_text'].append(span.content)

                            if span.index == sent.index:
                                text_features_dict['span_sent_pos'].append(sent.pos)
                                text_features_dict['span_sent_neg'].append(sent.neg)
                                text_features_dict['span_sent_neu'].append(sent.neu)

                elif output.transcription:
                    for span in output.transcription.subtitles:
                        if span.start.total_seconds() <= t <= span.end.total_seconds():
                            text_features_dict['frame'].append(i)
                            text_features_dict['span_start'].append(span.start.total_seconds())
                            text_features_dict['span_end'].append(span.end.total_seconds())
                            text_features_dict['span_text'].append(span.content)
                    
            audio_text_features_df = (pd.DataFrame(audio_annotation_dict).
                set_index('frame').
                merge(pd.DataFrame(text_features_dict).set_index('frame'), on=['frame'], how='left')
            )

            dfs.append(audio_text_features_df)

        if output.voice_features:
            dfs.append(pd.DataFrame(asdict(output.voice_features)).set_index('frame'))

        output.features = reduce(lambda left, right:
            pd.merge(left , right,
                on = ["frame"],
                how = "left"),
            dfs
        )


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
        output = Multimodal(filename=filepath)

        with VideoFileClip(filepath) as clip:
            self.audio_path = os.path.splitext(filepath)[0] + '.wav'
            subclip = clip.subclip(
                process_subclip[0], 
                process_subclip[1]
            )
            output.duration = subclip.duration
            output.fps = subclip.fps
            output.fps_adjusted = int(subclip.fps / skip_frames)
            self.time_step = 1/int(subclip.fps / skip_frames)

            if self.speaker_identifier or self.voice_extractor:
                # Use subclip if `process_subclip` is provided (default uses entire clip)
                subclip.audio.write_audiofile(self.audio_path)

        if self.face_extractor:
            video_annotation = self.face_extractor.apply(
                filepath,
                batch_size=frame_batch_size,
                skip_frames=skip_frames,
                process_subclip=process_subclip,
                show_progress=show_progress
            )
            output.video_annotation = video_annotation        

        if self.speaker_identifier:
            audio_annotation = self.speaker_identifier.apply(self.audio_path)

            output.audio_annotation = audio_annotation

            if self.audio_transcriber:
                transcription = self.audio_transcriber.apply(
                    self.audio_path,
                    audio_annotation=audio_annotation,
                    show_progress=show_progress
                )

                output.transcription = transcription

                if self.sentiment_extractor:
                    sentiment = self.sentiment_extractor.apply(
                        transcription=transcription,
                        show_progress=show_progress
                    )

                    output.sentiment = sentiment

        if self.voice_extractor:
            voice_features = self.voice_extractor.apply(
                self.audio_path,
                time_step=self.time_step
            )

            output.voice_features = voice_features
        
        self.merge_features(output=output)

        if not keep_audiofile and os.path.exists(self.audio_path):
            os.remove(self.audio_path)

        return output
