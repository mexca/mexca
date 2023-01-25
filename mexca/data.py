"""Objects for storing multimodal data.
"""

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from functools import reduce
from typing import Any, Dict, List, Optional, TextIO, Union
import numpy as np
import pandas as pd
import srt
from intervaltree import Interval, IntervalTree


@dataclass
class VideoAnnotation:
    """Video annotation class for storing facial features.

    Parameters
    ----------
    frame : list, optional
        Index of each frame.
    time : list, optional
        Timestamp of each frame in seconds.
    face_box : list, optional
        Bounding box of a detected face. Is `numpy.nan` if no face was detected.
    face_prob : list, optional
        Probability of a detected face. Is `numpy.nan` if no face was detected.
    face_landmarks : list, optional
        Facial landmarks of a detected face. Is `numpy.nan` if no face was detected.
    face_aus : list, optional
        Facial action unit activations of a detected face. Is `numpy.nan` if no face was detected.
    face_label : list, optional
        Label of a detected face. Is `numpy.nan` if no face was detected.
    face_confidence : list, optional
        Confidence of the `face_label` assignment. Is `numpy.nan` if no face was detected or
        only one face label was assigned.

    """
    frame: Optional[List[int]] = field(default_factory=list)
    time: Optional[List[float]] = field(default_factory=list)
    face_box: Optional[List[List[float]]] = field(default_factory=list)
    face_prob: Optional[List[float]] = field(default_factory=list)
    face_landmarks: Optional[List[List[List[float]]]] = field(default_factory=list)
    face_aus: Optional[List[List[float]]] = field(default_factory=list)
    face_label: Optional[List[Union[str, int]]] = field(default_factory=list)
    face_confidence: Optional[List[float]] = field(default_factory=list)


    @classmethod
    def _from_dict(cls, data: Dict):
        field_names = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


    @classmethod
    def from_json(cls, filename: str):
        """Load a video annotation from a JSON file.

        Parameters
        ----------
        filename: str
            Name of the JSON file from which the object should be loaded.
            Must have a .json ending.

        """
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls._from_dict(data=data)


    def write_json(self, filename: str):
        """Write the video annotation to a JSON file.

        Arguments
        ---------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file, allow_nan=True)


@dataclass
class VoiceFeatures:
    """Class for storing voice features.

    Features are stored as lists (like columns of a data frame).
    Optional features are initialized as empty lists.

    Parameters
    ----------
    frame: list
        The frame index for which features were extracted.
    time: list
        The time stamp at which features were extracted.
    pitch_f0: list, optional
        The voice pitch measured as the fundamental frequency F0.

    """
    frame: List[int]
    time: List[float]
    pitch_f0: Optional[List[float]] = field(default_factory=list)


    @classmethod
    def _from_dict(cls, data: Dict):
        field_names = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


    @classmethod
    def from_json(cls, filename: str):
        """Load voice features from a JSON file.

        Parameters
        ----------
        filename: str
            Name of the JSON file from which the object should be loaded.
            Must have a .json ending.

        """
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls._from_dict(data=data)


    def write_json(self, filename: str):
        """Store voice features in a JSON file.

        Arguments
        ---------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file, allow_nan=True)


def _get_rttm_header() -> List[str]:
    return ["type", "file", "chnl", "tbeg",
            "tdur", "ortho", "stype", "name",
            "conf"]


@dataclass
class SegmentData:
    """Class for storing speech segment data.

    Parameters
    ----------
    filename : str
        Name of the file from which the segment was obtained.
    channel : int
        Channel index.
    name : str, optional, default=None
        Speaker label.
    conf : float, optional, default=None
        Confidence of speaker label.

    """
    filename: str
    channel: int
    name: Optional[int] = None
    conf: Optional[float] = None


class SpeakerAnnotation(IntervalTree):
    """Class for storing speaker and speech segment annotations.

    Stores speech segments as ``intervaltree.Interval`` in an ``intervaltree.IntervalTree``.
    Speaker labels are stored in `SegmentData` objects in the `data` attribute of each interval.

    """
    def __init__(self, intervals: List[Interval] = None):
        super().__init__(intervals)


    def __str__(self, end: str = "\t", file: TextIO = sys.stdout, header: bool = True):
        if header:
            for h in _get_rttm_header():
                print(h, end=end, file=file)

            print("", file=file)

        for seg in self.items():
            for col in (
                "SPEAKER", seg.data.filename, seg.data.channel, seg.begin, seg.end-seg.begin,
                None, None, seg.data.name, seg.data.conf
            ):
                if col is not None:
                    if isinstance(col, float):
                        col = round(col, 2)
                    print(col, end=end, file=file)
                else:
                    print('<NA>', end=end, file=file)

            print("", file=file)

        return ""


    @classmethod
    def from_pyannote(cls, annotation: Any):
        """Create a `SpeakerAnnotation` object from a ``pyannote.core.Annotation`` object.

        Parameters
        ----------
        annotation : pyannote.core.Annotation
            Annotation object containing speech segments and speaker labels.

        """
        segments = []

        for seg, _, spk in annotation.itertracks(yield_label=True):
            segments.append(Interval(
                begin=seg.start,
                end=seg.end,
                data=SegmentData(
                    filename=annotation.uri,
                    channel=1,
                    name=str(spk)
                )
            ))

        return cls(intervals=segments)


    @classmethod
    def from_rttm(cls, filename: str):
        """Load a speaker annotation from an RTTM file.

        Parameters
        ----------
        filename : str
            Path to the file. Must have an RTTM ending.

        """
        with open(filename, "r", encoding='utf-8') as file:
            segments = []
            for row in file:
                row_split = [None if cell == "<NA>" else cell for cell in row.split(" ")]
                segment = Interval(
                    begin=float(row_split[3]),
                    end=float(row_split[3]) + float(row_split[4]),
                    data=SegmentData(
                        filename=row_split[1],
                        channel=int(row_split[2]),
                        name=row_split[7],
                    )
                )
                segments.append(segment)

            return cls(segments)


    def write_rttm(self, filename: str):
        """Write a speaker annotation to an RTTM file.

        Parameters
        ----------
        filename : str
            Path to the file. Must have an RTTM ending.

        """
        with open(filename, "w", encoding='utf-8') as file:
            self.__str__(end=" ", file=file, header=False) #pylint: disable=unnecessary-dunder-call


@dataclass
class TranscriptionData:
    """Class for storing transcription data.

    Parameters
    ----------
    index: int
        Index of the transcribed sentence.
    text: str
        Transcribed text.
    speaker: str, optional
        Speaker of the transcribed text.

    """
    index: int
    text: str
    speaker: Optional[str] = None


class AudioTranscription:
    """Class for storing audio transcriptions.

    Parameters
    ----------
    filename: str
        Name of the transcribed audio file.
    subtitles: intervaltree.IntervalTree, optional, default=None
        Interval tree containing the transcribed speech segments split into sentences as intervals.
        The transcribed sentences are stored in the `data` attribute of each interval.

    """
    def __init__(self,
        filename: str,
        subtitles: Optional[IntervalTree] = None
    ):
        self.filename = filename
        self.subtitles = subtitles


    def __len__(self) -> int:
        return len(self.subtitles)


    @classmethod
    def from_srt(cls, filename: str):
        """Load an audio transcription from an SRT file.

        Parameters
        ----------
        filename: str
            Name of the file to be loaded. Must have an .srt ending.

        """
        with open(filename, 'r', encoding='utf-8') as file:
            subtitles = srt.parse(file)

            intervals = []

            for sub in subtitles:
                content = sub.content.split('>')
                intervals.append(Interval(
                    begin=sub.start.total_seconds(),
                    end=sub.end.total_seconds(),
                    data=TranscriptionData(
                        index=sub.index,
                        text=content[1],
                        speaker=content[0][1:]
                    )
                ))

            return cls(filename=filename, subtitles=IntervalTree(intervals))


    def write_srt(self, filename: str):
        """Write an audio transcription to an SRT file

        Parameters
        ----------
        filename: str
            Name of the file to write to. Must have an .srt ending.

        """
        subtitles = []

        for iv in self.subtitles.all_intervals:
            content = f"<{iv.data.speaker}> {iv.data.text}"
            subtitles.append(srt.Subtitle(
                index=iv.data.index,
                start=timedelta(seconds=iv.begin),
                end=timedelta(seconds=iv.end),
                content=content
            ))

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(srt.compose(subtitles))


@dataclass
class SentimentData:
    """Class for storing sentiment data.

    Parameters
    ----------
    index: int
        Index of the sentence for which sentiment scores were predicted.
    pos: float
        Positive sentiment score.
    neg: float
        Negative sentiment score.
    neu: float
        Neutral sentiment score.

    """
    index: int
    pos: float
    neg: float
    neu: float


@dataclass
class SentimentAnnotation(IntervalTree):
    """Class for storing sentiment scores of transcribed sentences.

    Stores sentiment scores as intervals in an interval tree. The scores are stored in the `data` attribute of each interval.

    """
    def __init__(self, intervals: List[Interval] = None):
        super().__init__(intervals)


    @classmethod
    def from_json(cls, filename: str):
        """Load a sentiment annotation from a JSON file.

        Parameters
        ----------
        filename: str
            Name of the JSON file from which the object should be loaded.
            Must have a .json ending.

        """
        with open(filename, 'r', encoding='utf-8') as file:
            sentiment = json.load(file)

            intervals = []

            for sen in sentiment:
                intervals.append(Interval(
                    begin=sen['begin'],
                    end=sen['end'],
                    data=SentimentData(
                        index=sen['index'],
                        pos=sen['pos'],
                        neg=sen['neg'],
                        neu=sen['neu']
                    )
                ))

            return cls(intervals=intervals)


    def write_json(self, filename: str):
        """Write a sentiment annotation to a JSON file.

        Parameters
        ----------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, 'w', encoding='utf-8') as file:
            sentiment = []

            for iv in self.all_intervals:
                data_dict = asdict(iv.data)
                data_dict['begin'] = iv.begin
                data_dict['end'] = iv.end
                sentiment.append(data_dict)

            json.dump(sentiment, file, allow_nan=True)


class Multimodal:
    """Class for storing multimodal features.

    See the :ref:`Output` section for details.

    Parameters
    ----------
    filename : str
        Name of the file from which features were extracted.
    duration : float, optional, default=None
        Video duration in seconds.
    fps: : float
        Frames per second.
    fps_adjusted : float
        Frames per seconds adjusted for skipped frames.
        Mostly needed for internal computations.
    video_annotation : VideoAnnotation
        Object containing facial features.
    audio_annotation : SpeakerAnnotation
        Object containing speech segments and speakers.
    voice_features : VoiceFeatures
        Object containing voice features.
    transcription : AudioTranscription
        Object containing transcribed speech segments split into sentences.
    sentiment : SentimentAnnotation
        Object containing sentiment scores for transcribed sentences.
    features : pandas.DataFrame
        Merged features.

    """
    def __init__(self,
        filename: str,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        fps_adjusted: Optional[int] = None,
        video_annotation: Optional[VideoAnnotation] = None,
        audio_annotation: Optional[SpeakerAnnotation] = None,
        voice_features: Optional[VoiceFeatures] = None,
        transcription: Optional[AudioTranscription] = None,
        sentiment: Optional[SentimentAnnotation] = None,
        features: Optional[pd.DataFrame] = None
    ):
        self.filename = filename
        self.duration = duration
        self.fps = fps
        self.fps_adjusted = fps if fps_adjusted is None else fps_adjusted
        self.video_annotation = video_annotation
        self.audio_annotation = audio_annotation
        self.voice_features = voice_features
        self.transcription = transcription
        self.sentiment = sentiment
        self.features = features


    def _merge_video_annotation(self, data_frames: List):
        if self.video_annotation:
            data_frames.append(pd.DataFrame(asdict(self.video_annotation)))


    def _merge_audio_text_features(self, data_frames: List):
        if self.audio_annotation:
            audio_annotation_dict = {
                "frame": [],
                "segment_start": [],
                "segment_end": [],
                "segment_speaker_label": []
            }

            time = np.arange(0.0, self.duration, 1/self.fps_adjusted, dtype=np.float32)
            frame = np.arange(0, self.duration*self.fps, self.fps/self.fps_adjusted, dtype=np.int32)

            if self.transcription:
                text_features_dict = {
                    "frame": [],
                    "span_start": [],
                    "span_end": [],
                    "span_text": [],
                    "segment_speaker_label": []
                }

                if self.sentiment:
                    text_features_dict['span_sent_pos'] = []
                    text_features_dict['span_sent_neg'] = []
                    text_features_dict['span_sent_neu'] = []

            for i, t in zip(frame, time):
                overlap_segments = self.audio_annotation[t]

                if len(overlap_segments) > 0:
                    for seg in overlap_segments:
                        audio_annotation_dict['frame'].append(i)
                        audio_annotation_dict['segment_start'].append(seg.begin)
                        audio_annotation_dict['segment_end'].append(seg.end)
                        audio_annotation_dict['segment_speaker_label'].append(str(seg.data.name))
                else:
                    audio_annotation_dict['frame'].append(i)
                    audio_annotation_dict['segment_start'].append(np.NaN)
                    audio_annotation_dict['segment_end'].append(np.NaN)
                    audio_annotation_dict['segment_speaker_label'].append(np.NaN)

                if self.transcription and self.sentiment:
                    for span, sent in zip(self.transcription.subtitles[t], self.sentiment[t]):
                        text_features_dict['frame'].append(i)
                        text_features_dict['span_start'].append(span.begin)
                        text_features_dict['span_end'].append(span.end)
                        text_features_dict['span_text'].append(span.data.text)
                        text_features_dict['segment_speaker_label'].append(str(span.data.speaker))

                        if span.data.index == sent.data.index:
                            text_features_dict['span_sent_pos'].append(sent.data.pos)
                            text_features_dict['span_sent_neg'].append(sent.data.neg)
                            text_features_dict['span_sent_neu'].append(sent.data.neu)

                elif self.transcription:
                    for span in self.transcription.subtitles[t]:      
                        text_features_dict['frame'].append(i)
                        text_features_dict['span_start'].append(span.begin)
                        text_features_dict['span_end'].append(span.end)
                        text_features_dict['span_text'].append(span.data.text)
                        text_features_dict['segment_speaker_label'].append(span.data.speaker)
                    
            audio_text_features_df = pd.DataFrame(audio_annotation_dict)

            if self.transcription:
                audio_text_features_df = audio_text_features_df.merge(
                    pd.DataFrame(text_features_dict),
                    on=['frame', 'segment_speaker_label'],
                    how='left'
                )

            data_frames.append(audio_text_features_df)


    def _merge_voice_features(self, data_frames: List):
        if self.voice_features:
            data_frames.append(pd.DataFrame(asdict(self.voice_features)))


    @staticmethod
    def _delete_time_col(df: pd.DataFrame) -> pd.DataFrame:
        if 'time' in df.columns:
            del df['time']
        return df


    def merge_features(self) -> pd.DataFrame:
        """Merge multimodal features from pipeline components into a common data frame.

        Transforms and merges the available output stored in the `Multimodal` object
        based on the `'frame'` variable. Stores the merged features as a `pandas.DataFrame`
        in the `features` attribute.

        Returns
        -------
        pandas.DataFrame
            Merged multimodal features.

        """

        dfs = []

        self._merge_video_annotation(data_frames=dfs)

        self._merge_audio_text_features(data_frames=dfs)

        self._merge_voice_features(data_frames=dfs)

        if len(dfs) > 0:
            dfs = map(self._delete_time_col, dfs)
            self.features = reduce(lambda left, right:
                pd.merge(left , right,
                    on = ["frame"],
                    how = "left"),
                dfs
            )

            time = self.features.frame * (1/self.fps)

            self.features.insert(1, 'time', time)

        return self.features
