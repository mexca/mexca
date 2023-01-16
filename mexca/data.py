"""Create objects for storing multimodal data
"""

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from functools import reduce
from typing import Any, Dict, List, Optional, TextIO, Union
import srt
import numpy as np
import pandas as pd


@dataclass
class VideoAnnotation:
    """Video annotation class for storing facial features and video meta data.

    Parameters
    ----------
    filename : str
        Name of the video file.
    duration : float
        Duration of the video.
    fps : int
        Frames per second of the video.
    frame : list
        Index of each frame.
    time : list
        Timestamp of each frame in seconds.
    face_box : list
        Bounding box of a detected face. Is `numpy.nan` if no face was detected.
    face_prob : list
        Probability of a detected face. Is `numpy.nan` if no face was detected.
    face_landmarks : list
        Facial landmarks of a detected face. Is `numpy.nan` if no face was detected.
    face_aus : list
        Facial action unit activations of a detected face. Is `numpy.nan` if no face was detected.
    face_label : list
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
    def from_dict(cls, data: Dict):
        field_names = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


    @classmethod
    def from_json(cls, filename: str):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls.from_dict(data=data)


    def write_json(self, filename: str):
        """Store the video annotation in a json file.

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
    """
    frame: List[int]
    pitch_F0: Optional[List[float]] = field(default_factory=list)


    @classmethod
    def from_dict(cls, data: Dict):
        field_names = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


    @classmethod
    def from_json(cls, filename: str):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls.from_dict(data=data)


    def write_json(self, filename: str):
        """Store voice features in a json file.

        Arguments
        ---------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file, allow_nan=True)


@dataclass
class RttmSegment:
    type: str
    file: str
    chnl: int
    tbeg: float
    tdur: float
    ortho: Optional[str] = None
    stype: Optional[str] = None
    name: Optional[str] = None
    conf: Optional[float] = None


def _get_rttm_header() -> List[str]:
    return ["type", "file", "chnl", "tbeg", 
            "tdur", "ortho", "stype", "name", 
            "conf"]


@dataclass
class RttmAnnotation:
    segments: List[RttmSegment]
    header: List[str] = field(default_factory=_get_rttm_header)


    def __str__(self, end: str = "\t", file: TextIO = sys.stdout, header: bool = True):
        if header:
            for h in self.header:
                print(h, end=end, file=file)

            print("", file=file)

        for seg in self.segments:
            for _, value in seg.__dict__.items():
                if isinstance(value, type(None)):
                    print("<NA>", end=end, file=file)
                elif isinstance(value, float):
                    print(round(value, 2), end=end, file=file)
                else:
                    print(str(value), end=end, file=file)

            print("", file=file)

        return ""


    @classmethod
    def from_pyannote(cls, annotation: Any):
        segments = []

        for seg, _, spk in annotation.itertracks(yield_label=True):
            segments.append(RttmSegment(
                type='SPEAKER',
                file=annotation.uri,
                chnl=1,
                tbeg=seg.start,
                tdur=seg.duration,
                name=spk
            ))

        return cls(segments)


    @classmethod
    def from_rttm(cls, filename: str):
        with open(filename, "r", encoding='utf-8') as file:
            segments = []
            for row in file:
                row_split = [None if cell == "<NA>" else cell for cell in row.split(" ")]
                segment = RttmSegment(
                    type=row_split[0],
                    file=row_split[1],
                    chnl=int(row_split[2]),
                    tbeg=float(row_split[3]),
                    tdur=float(row_split[4]),
                    ortho=row_split[5],
                    stype=row_split[6],
                    name=row_split[7],
                    conf=float(row_split[8]) if row_split[8] is not None else None
                )
                segments.append(segment)

            return cls(segments)


    def write_rttm(self, filename: str):
        with open(filename, "w", encoding='utf-8') as file:
            self.__str__(end=" ", file=file, header=False)


@dataclass
class AudioTranscription:
    filename: str
    subtitles: List[srt.Subtitle] = field(default_factory=list)


    def __len__(self) -> int:
        return len(self.subtitles)


    @classmethod
    def from_srt(cls, filename: str):
        with open(filename, 'r', encoding='utf-8') as file:
            subtitles = srt.parse(file)

            return cls(filename=filename, subtitles=list(subtitles))


    def write_srt(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(srt.compose(self.subtitles))


@dataclass
class Sentiment:
    index: int
    pos: float
    neg: float
    neu: float


@dataclass
class SentimentAnnotation:
    sentiment: List[Sentiment] = field(default_factory=list)


    @classmethod
    def from_dict(cls, data: Dict):
        field_names = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        filtered_data['sentiment'] = [Sentiment(
            index=s['index'],
            pos=s['pos'],
            neg=s['neg'],
            neu=s['neu']
        ) for s in filtered_data['sentiment']]
        return cls(**filtered_data)


    @classmethod
    def from_json(cls, filename: str):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls.from_dict(data=data)


    def write_json(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file, allow_nan=True)


class Multimodal:
    def __init__(self,
        filename: str,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        fps_adjusted: Optional[int] = None,
        video_annotation: Optional[VideoAnnotation] = None,
        audio_annotation: Optional[RttmAnnotation] = None,
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
            data_frames.append(pd.DataFrame(asdict(self.video_annotation)).set_index('frame'))


    def _merge_audio_text_features(self, data_frames: List):
        if self.audio_annotation: #pylint: disable=too-many-nested-blocks
            audio_annotation_dict = {
                "frame": [],
                "segment_start": [],
                "segment_end": [],
                "segment_speaker_label": []
            }

            time = np.arange(0.0, self.duration, 1/self.fps_adjusted, dtype=np.float32)
            frame = np.arange(0, self.duration*self.fps, self.fps_adjusted, dtype=np.int32)

            if self.transcription:
                text_features_dict = {
                    "frame": [],
                    "span_start": [],
                    "span_end": [],
                    "span_text": []
                }

                if self.sentiment:
                    text_features_dict['span_sent_pos'] = []
                    text_features_dict['span_sent_neg'] = []
                    text_features_dict['span_sent_neu'] = []

            for i, t in zip(frame, time):
                for seg in self.audio_annotation.segments:
                    seg_end = seg.tbeg + seg.tdur
                    if seg.tbeg <= t <= seg_end:
                        audio_annotation_dict['frame'].append(i)
                        audio_annotation_dict['segment_start'].append(seg.tbeg)
                        audio_annotation_dict['segment_end'].append(seg_end)
                        audio_annotation_dict['segment_speaker_label'].append(seg.name)

                if self.transcription and self.sentiment:
                    for span, sent in zip(self.transcription.subtitles, self.sentiment.sentiment):
                        if span.start.total_seconds() <= t <= span.end.total_seconds():
                            text_features_dict['frame'].append(i)
                            text_features_dict['span_start'].append(span.start.total_seconds())
                            text_features_dict['span_end'].append(span.end.total_seconds())
                            text_features_dict['span_text'].append(span.content)

                            if span.index == sent.index:
                                text_features_dict['span_sent_pos'].append(sent.pos)
                                text_features_dict['span_sent_neg'].append(sent.neg)
                                text_features_dict['span_sent_neu'].append(sent.neu)

                elif self.transcription:
                    for span in self.transcription.subtitles:
                        if span.start.total_seconds() <= t <= span.end.total_seconds():
                            text_features_dict['frame'].append(i)
                            text_features_dict['span_start'].append(span.start.total_seconds())
                            text_features_dict['span_end'].append(span.end.total_seconds())
                            text_features_dict['span_text'].append(span.content)
                    
            audio_text_features_df = (pd.DataFrame(audio_annotation_dict).
                set_index('frame').
                merge(pd.DataFrame(text_features_dict).set_index('frame'), on=['frame'], how='left')
            )

            data_frames.append(audio_text_features_df)


    def _merge_voice_features(self, data_frames: List):
        if self.voice_features:
            data_frames.append(pd.DataFrame(asdict(self.voice_features)).set_index('frame'))


    def merge_features(self):
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
            self.features = reduce(lambda left, right:
                pd.merge(left , right,
                    on = ["frame"],
                    how = "left"),
                dfs
            ).reset_index()

        return self.features
