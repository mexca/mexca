
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, TextIO, Union
import srt

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
        sentiment: Optional[SentimentAnnotation] = None
    ):
        self.filename = filename
        self.duration = duration
        self.fps = fps
        self.fps_adjusted = fps_adjusted
        self.video_annotation = video_annotation
        self.audio_annotation = audio_annotation
        self.voice_features = voice_features
        self.transcription = transcription
        self.sentiment = sentiment
