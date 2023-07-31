"""Objects for storing multimodal data.
"""

import json
import sys
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import reduce
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import srt
import yaml
from intervaltree import Interval, IntervalTree
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    InstanceOf,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

EMPTY_VALUE = None
"""Value that is returned if a feature is not present.
"""

_ProbFloat = Annotated[Optional[NonNegativeFloat], Field(le=1.0)]


def _float2str(x: Union[Optional[float], Optional[str]]) -> Optional[str]:
    if isinstance(x, str):
        return x
    if isinstance(x, (float, int)):
        return str(int(x))
    return None


_Float2Str = Annotated[
    Optional[str],
    BeforeValidator(_float2str),
]


def _check_sorted(x: List):
    if x == sorted(x):
        return x
    raise ValueError("Attribute must be in ascending order")


def _check_common_length(obj: BaseModel) -> Any:
    for v in obj.model_fields:
        a = getattr(obj, v)
        if isinstance(a, list) and len(a) > 0 and len(a) != len(obj.frame):
            raise ValueError(
                f"List attribute {v} must have the same length as 'frame'"
            )

    return obj


# Adapted from librosa package: https://github.com/librosa/librosa/blob/main/librosa/_typing.py
_Window = Union[str, Tuple[Any, ...], float]


class BaseData(BaseModel, ABC):
    """Base class for storing segment data."""


class BaseFeatures(BaseModel, ABC):
    """Base class for storing features.

    Parameters
    ----------
    filename: pydantic.FilePath
        Path to the video file. Must be a valid path.
    """

    filename: FilePath

    @classmethod
    def from_json(
        cls,
        filename: str,
        extra_filename: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        """Load data from a JSON file.

        Parameters
        ----------
        filename: str
            Name of the JSON file from which the object should be loaded.
            Must have a .json ending.

        """
        with open(filename, "r", encoding=encoding) as file:
            data = json.load(file)

        if extra_filename is not None:
            data["filename"] = extra_filename

        return cls(**data)

    def write_json(self, filename: str, encoding: str = "utf-8"):
        """Store data in a JSON file.

        Arguments
        ---------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, "w", encoding=encoding) as file:
            file.write(self.model_dump_json())


class BaseAnnotation(BaseModel, ABC):
    """Base class for storing annotations."""

    filename: FilePath
    segments: Optional[InstanceOf[IntervalTree]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    @abstractmethod
    def data_type() -> Any:
        pass

    @classmethod
    def from_json(
        cls,
        filename: str,
        extra_filename: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        """Load data from a JSON file.

        Parameters
        ----------
        filename: str
            Name of the JSON file from which the object should be loaded.
            Must have a .json ending.

        """
        with open(filename, "r", encoding=encoding) as file:
            data = json.load(file)

            segments = [
                Interval(
                    begin=seg["begin"],
                    end=seg["end"],
                    data=cls.data_type().model_validate(seg["data"]),
                )
                for seg in data
            ]

            return cls(
                filename=filename if extra_filename is None else extra_filename,
                segments=IntervalTree(segments),
            )

    def write_json(self, filename: str, encoding: str = "utf-8"):
        """Store data in a JSON file.

        Arguments
        ---------
        filename: str
            Name of the destination file. Must have a .json ending.

        """
        with open(filename, "w", encoding=encoding) as file:
            data = [
                {"begin": iv.begin, "end": iv.end, "data": iv.data.model_dump()}
                for iv in self.segments.all_intervals
            ]

            json.dump(data, file)


class VideoAnnotation(BaseFeatures):
    """Video annotation class for storing facial features.

    Parameters
    ----------
    frame : list
        Index of each frame.
    time : list
        Timestamp of each frame in seconds.
    face_box : list, optional
        Bounding box of a detected face. Is `None` if no face was detected.
    face_prob : list, optional
        Probability of a detected face. Is `None` if no face was detected.
    face_landmarks : list, optional
        Facial landmarks of a detected face. Is `None` if no face was detected.
    face_aus : list, optional
        Facial action unit activations of a detected face. Is `None` if no face was detected.
    face_label : list, optional
        Label of a detected face. Is `None` if no face was detected.
    face_confidence : list, optional
        Confidence of the `face_label` assignment. Is `None` if no face was detected or
        only one face label was assigned.
    face_average_embeddings : dict, optional
        Average embedding vector (list of 512 float elements) for each face in the input video.
    """

    frame: List[NonNegativeInt] = Field(default_factory=list)
    time: List[NonNegativeFloat] = Field(default_factory=list)
    face_box: Optional[List[Optional[List[NonNegativeFloat]]]] = Field(
        default_factory=list
    )
    face_prob: Optional[List[_ProbFloat]] = Field(default_factory=list)
    face_landmarks: Optional[
        List[Optional[List[List[NonNegativeFloat]]]]
    ] = Field(default_factory=list)
    face_aus: Optional[List[Optional[List[_ProbFloat]]]] = Field(
        default_factory=list
    )
    face_label: Optional[List[_Float2Str]] = Field(default_factory=list)
    face_confidence: Optional[List[_ProbFloat]] = Field(default_factory=list)
    face_average_embeddings: Optional[Dict[_Float2Str, List[float]]] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(validate_assignment=True)

    _check_sorted_frame = field_validator("frame", mode="after")(_check_sorted)
    _check_sorted_time = field_validator("time", mode="after")(_check_sorted)

    @field_validator("face_box", mode="after")
    def _check_len_face_box(cls, v):
        if v is not None and any(len(e) != 4 for e in v if e is not None):
            raise ValueError("All face boxes must have four coordinates")
        return v

    @field_validator("face_landmarks", mode="after")
    def _check_len_face_landmarks(cls, v):
        if v is not None and any(
            len(b) != 2 for e in v if e is not None for b in e
        ):
            raise ValueError(
                "All face landmarks must have x and y coordinate pairs"
            )
        return v

    @model_validator(mode="after")
    def _check_finite(self) -> "VideoAnnotation":
        for frm, box, prob, lmk, au, label in zip(
            self.frame,
            self.face_box,
            self.face_prob,
            self.face_landmarks,
            self.face_aus,
            self.face_label,
        ):
            if box is None and not (box == prob == lmk == au == label):
                raise ValueError(
                    f"Face boxes, probabilities, landmarks, action units, and labels not all valid or invalid for frame {frm}"
                )

        return self

    _common_length = model_validator(mode="after")(_check_common_length)

    @model_validator(mode="after")
    def _check_face_labels(self) -> "VideoAnnotation":
        if not self.face_average_embeddings or not self.face_label:
            return self
        unique_labels = set(self.face_label)

        if all(
            lbl in self.face_average_embeddings.keys()
            for lbl in unique_labels
            if lbl is not None
        ):
            return self
        raise ValueError(
            f"Keys in 'face_average_embeddings' {self.face_average_embeddings.keys()} must be the same as unique values in 'face_label' {unique_labels}"
        )


class VoiceFeaturesConfig(BaseModel):
    """Configure the calculation of signal properties used for voice feature extraction.

    Create a pseudo-immutable object with attributes that are recognized by the
    :class:`VoiceExtractor` class and forwarded as arguments to signal property objects defined
    in :mod:`mexca.audio.features`. Details can be found in the feature class documentation.

    Parameters
    ----------
    frame_len: int
        Number of samples per frame.
    hop_len: int
        Number of samples between frame starting points.
    center: bool, default=True
        Whether the signal has been centered and padded before framing.
    pad_mode: str, default='constant'
        How the signal has been padded before framing. See :func:`numpy.pad`.
        Uses the default value 0 for `'constant'` padding.
    spec_window: str or float or tuple, default="hann"
        The window that is applied before the STFT to obtain spectra.
    pitch_lower_freq: float, default=75.0
        Lower limit used for pitch estimation (in Hz).
    pitch_upper_freq: float, default=600.0
        Upper limit used for pitch estimation (in Hz).
    pitch_method: str, default="pyin"
        Method used for estimating voice pitch.
    ptich_n_harmonics: int, default=100
        Number of estimated pitch harmonics.
    pitch_pulse_lower_period: float, optional, default=0.0001
        Lower limit for periods between glottal pulses for jitter and shimmer extraction.
    pitch_pulse_upper_period: float, optional, default=0.02
        Upper limit for periods between glottal pulses for jitter and shimmer extraction.
    pitch_pulse_max_period_ratio: float, optional, default=1.3
        Maximum ratio between consecutive glottal periods for jitter and shimmer extraction.
    pitch_pulse_max_amp_factor: float, default=1.6
        Maximum ratio between consecutive amplitudes used for shimmer extraction.
    jitter_rel: bool, default=True
        Divide jitter by the average pitch period.
    shimmer_rel: bool, default=True
        Divide shimmer by the average pulse amplitude.
    hnr_lower_freq: float, default = 75.0
        Lower fundamental frequency limit for choosing pitch candidates when computing the harmonics-to-noise ratio (HNR).
    hnr_rel_silence_threshold: float, default = 0.1
        Relative threshold for treating signal frames as silent when computing the HNR.
    formants_max: int, default=5
        The maximum number of formants that are extracted.
    formants_lower_freq: float, default=50.0
        Lower limit for formant frequencies (in Hz).
    formants_upper_freq: float, default=5450.0
        Upper limit for formant frequencies (in Hz).
    formants_signal_preemphasis_from: float, default=50.0
        Starting value for the applied preemphasis function (in Hz).
    formants_window: str or float or tuple, default="praat_gaussian"
        Window function that is applied before formant estimation.
    formants_amp_lower: float, optional, default=0.8
        Lower boundary for formant peak amplitude search interval.
    formants_amp_upper: float, optional, default=1.2
        Upper boundary for formant peak amplitude search interval.
    formants_amp_rel_f0: bool, optional, default=True
        Whether the formant amplitude is divided by the fundamental frequency amplitude.
    alpha_ratio_lower_band: tuple, default=(50.0, 1000.0)
        Boundaries of the alpha ratio lower frequency band (start, end) in Hz.
    alpha_ratio_upper_band: tuple, default=(1000.0, 5000.0)
        Boundaries of the alpha ratio upper frequency band (start, end) in Hz.
    hammar_index_pivot_point_freq: float, default=2000.0
        Point separating the Hammarberg index lower and upper frequency regions in Hz.
    hammar_index_upper_freq: float, default=5000.0
        Upper limit for the Hammarberg index upper frequency region in Hz.
    spectral_slopes_bands: tuple, default=((0.0, 500.0), (500.0, 1500.0))
        Frequency bands in Hz for which spectral slopes are estimated.
    mel_spec_n_mels: int, default=26
        Number of Mel filters.
    mel_spec_lower_freq: float, default=20.0
        Lower frequency boundary for Mel spectogram transformation in Hz.
    mel_spec_upper_freq: float, default=8000.0
        Upper frequency boundary for Mel spectogram transformation in Hz.
    mfcc_n: int, default=4
        Number of Mel frequency cepstral coefficients (MFCCs) that are estimated per frame.
    mfcc_lifter: float, default=22.0
        Cepstral liftering coefficient for MFCC estimation. Must be >= 0. If zero, no liftering is applied.

    """

    frame_len: PositiveInt = 1024
    hop_len: PositiveInt = 256
    center: bool = True
    pad_mode: str = "constant"
    spec_window: _Window = "hann"
    pitch_lower_freq: NonNegativeFloat = 75.0
    pitch_upper_freq: NonNegativeFloat = 600.0
    pitch_method: str = "pyin"
    pitch_n_harmonics: PositiveInt = 100
    pitch_pulse_lower_period: PositiveFloat = 0.0001
    pitch_pulse_upper_period: PositiveFloat = 0.02
    pitch_pulse_max_period_ratio: PositiveFloat = 1.3
    pitch_pulse_max_amp_factor: PositiveFloat = 1.6
    jitter_rel: bool = True
    shimmer_rel: bool = True
    hnr_lower_freq: PositiveFloat = 75.0
    hnr_rel_silence_threshold: PositiveFloat = 0.1
    formants_max: PositiveInt = 5
    formants_lower_freq: NonNegativeFloat = 50.0
    formants_upper_freq: NonNegativeFloat = 5450.0
    formants_signal_preemphasis_from: Optional[NonNegativeFloat] = None
    formants_window: _Window = "praat_gaussian"
    formants_amp_lower: PositiveFloat = 0.8
    formants_amp_upper: PositiveFloat = 1.2
    formants_amp_rel_f0: bool = True
    alpha_ratio_lower_band: Tuple[NonNegativeFloat, NonNegativeFloat] = (
        50.0,
        1000.0,
    )
    alpha_ratio_upper_band: Tuple[NonNegativeFloat, NonNegativeFloat] = (
        1000.0,
        5000.0,
    )
    hammar_index_pivot_point_freq: PositiveFloat = 2000.0
    hammar_index_upper_freq: PositiveFloat = 5000.0
    spectral_slopes_bands: Tuple[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Tuple[NonNegativeFloat, NonNegativeFloat],
    ] = ((0.0, 500.0), (500.0, 1500.0))
    mel_spec_n_mels: PositiveInt = 26
    mel_spec_lower_freq: NonNegativeFloat = 20.0
    mel_spec_upper_freq: NonNegativeFloat = 8000.0
    mfcc_n: PositiveInt = 4
    mfcc_lifter: PositiveInt = 22

    @classmethod
    def from_yaml(cls, filename: str):
        """Load a voice configuration object from a YAML file.

        Uses safe YAML loading (only supports native YAML but no Python tags).
        Converts loaded YAML sequences to tuples.

        Parameters
        ----------
        filename: str
            Path to the YAML file. Must have a .yml or .yaml ending.

        """
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)

        return cls(**config_dict)

    def write_yaml(self, filename: str):
        """Write a voice configuration object to a YAML file.

        Uses safe YAML dumping (only supports native YAML but no Python tags).

        Parameters
        ----------
        filename: str
            Path to the YAML file. Must have a .yml or .yaml ending.

        """
        with open(filename, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(), file)


class VoiceFeatures(BaseFeatures):
    """Class for storing voice features.

    Features are stored as lists (like columns of a data frame).
    Optional features are initialized as empty lists.

    Parameters
    ----------
    frame: list
        The frame index for which features were extracted.
    time: list
        The time stamp at which features were extracted.

    """

    frame: List[NonNegativeInt]
    time: List[NonNegativeFloat]

    model_config = ConfigDict(validate_assignment=True)

    _check_sorted_frame = field_validator("frame", mode="after")(_check_sorted)
    _check_sorted_time = field_validator("time", mode="after")(_check_sorted)

    _common_length = model_validator(mode="after")(_check_common_length)

    def add_feature(self, name: str, feature: List[float]):
        self.__class__ = create_model(
            "VoiceFeatures",
            **{name: (List[float], Field(default_factory=list))},
            __base__=(self.__class__,),
        )
        setattr(self, name, feature)


def _get_rttm_header() -> List[str]:
    return [
        "type",
        "file",
        "chnl",
        "tbeg",
        "tdur",
        "ortho",
        "stype",
        "name",
        "conf",
    ]


class SegmentData(BaseData):
    """Class for storing speech segment data.

    Parameters
    ----------
    name : str
        Speaker label.
    conf : float, optional, default=None
        Confidence of speaker label.

    """

    name: str
    conf: Optional[_ProbFloat] = None


class SpeakerAnnotation(BaseAnnotation):
    """Class for storing speaker and speech segment annotations.

    Parameters
    ----------
    filename : str, optional
        Name of the audio file which is annotated.
    channel : int, optional
        Channel index.
    segments : intervaltree.IntervalTree, optional
        Stores speech segments as :class:`intervaltree.Interval`.
        Speaker labels are stored in :class:`SegmentData` objects in the :class:`data` attribute of each interval.

    """

    channel: Optional[int] = None

    @staticmethod
    def data_type() -> Any:
        return SegmentData

    def __str__(
        self, end: str = "\t", file: TextIO = sys.stdout, header: bool = True
    ):
        if header:
            for h in _get_rttm_header():
                print(h, end=end, file=file)

            print("", file=file)

        for seg in self.segments.items():
            for col in (
                "SPEAKER",
                self.filename,
                self.channel,
                seg.begin,
                seg.end - seg.begin,
                None,
                None,
                seg.data.name,
                seg.data.conf,
            ):
                if col is not None:
                    if isinstance(col, float):
                        col = round(col, 2)
                    print(col, end=end, file=file)
                else:
                    print("<NA>", end=end, file=file)

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
            segments.append(
                Interval(
                    begin=seg.start,
                    end=seg.end,
                    data=SegmentData(name=str(spk)),
                )
            )

        return cls(
            filename=annotation.uri, channel=1, segments=IntervalTree(segments)
        )

    @classmethod
    def from_rttm(cls, filename: str, extra_filename: Optional[str] = None):
        """Load a speaker annotation from an RTTM file.

        Parameters
        ----------
        filename : str
            Path to the file. Must have an RTTM ending.

        """
        with open(filename, "r", encoding="utf-8") as file:
            segments = []
            for row in file:
                row_split = [
                    None if cell == "<NA>" else cell for cell in row.split(" ")
                ]
                segment = Interval(
                    begin=float(row_split[3]),
                    end=float(row_split[3]) + float(row_split[4]),
                    data=SegmentData(
                        name=row_split[7],
                    ),
                )
                segments.append(segment)

            return cls(
                filename=row_split[1]
                if extra_filename is None
                else extra_filename,
                channel=int(row_split[2]),
                segments=IntervalTree(segments),
            )

    # pylint: disable=unnecessary-dunder-call
    def write_rttm(self, filename: str):
        """Write a speaker annotation to an RTTM file.

        Parameters
        ----------
        filename : str
            Path to the file. Must have an RTTM ending.

        """
        with open(filename, "w", encoding="utf-8") as file:
            self.__str__(end=" ", file=file, header=False)


class TranscriptionData(BaseData):
    """Class for storing transcription data.

    Parameters
    ----------
    index: int
        Index of the transcribed sentence.
    text: str
        Transcribed text.
    speaker: str, optional, default=None
        Speaker of the transcribed text.
    confidence : float, optional, default=None
        Average word probability of transcribed text.

    """

    index: int
    text: str
    speaker: Optional[str] = None
    confidence: Optional[_ProbFloat] = None


class AudioTranscription(BaseAnnotation):
    """Class for storing audio transcriptions.

    Parameters
    ----------
    filename: str
        Name of the transcribed audio file.
    segments: intervaltree.IntervalTree, optional, default=None
        Interval tree containing the transcribed speech segments split into sentences as intervals.
        The transcribed sentences are stored in the `data` attribute of each interval.

    """

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def subtitles(self):
        """Deprecated alias for `segments`."""
        return self.segments

    @staticmethod
    def data_type() -> Any:
        return TranscriptionData

    @classmethod
    def from_srt(cls, filename: str, extra_filename: Optional[str] = None):
        """Load an audio transcription from an SRT file.

        Parameters
        ----------
        filename: str
            Name of the file to be loaded. Must have an .srt ending.

        """
        with open(filename, "r", encoding="utf-8") as file:
            segments = srt.parse(file)

            intervals = []

            for sub in segments:
                content = sub.content.split(">")
                intervals.append(
                    Interval(
                        begin=sub.start.total_seconds(),
                        end=sub.end.total_seconds(),
                        data=TranscriptionData(
                            index=sub.index,
                            text=content[1],
                            speaker=content[0][1:],
                        ),
                    )
                )

            return cls(
                filename=filename if extra_filename is None else extra_filename,
                segments=IntervalTree(intervals),
            )

    def write_srt(self, filename: str):
        """Write an audio transcription to an SRT file

        Parameters
        ----------
        filename: str
            Name of the file to write to. Must have an .srt ending.

        """
        segments = []

        for iv in self.segments.all_intervals:
            content = f"<{iv.data.speaker}> {iv.data.text}"
            segments.append(
                srt.Subtitle(
                    index=iv.data.index,
                    start=timedelta(seconds=iv.begin),
                    end=timedelta(seconds=iv.end),
                    content=content,
                )
            )

        with open(filename, "w", encoding="utf-8") as file:
            file.write(srt.compose(segments))


class SentimentData(BaseData):
    """Class for storing sentiment data.

    Parameters
    ----------
    text: str
        Text of the sentence for which sentiment scores were predicted.
    pos: float
        Positive sentiment score.
    neg: float
        Negative sentiment score.
    neu: float
        Neutral sentiment score.

    """

    text: str
    pos: _ProbFloat
    neg: _ProbFloat
    neu: _ProbFloat


class SentimentAnnotation(BaseAnnotation):
    """Class for storing sentiment scores of transcribed sentences.

    Stores sentiment scores as intervals in an interval tree. The scores are stored in the `data` attribute of each interval.

    """

    @staticmethod
    def data_type() -> Any:
        return SentimentData


class Multimodal(BaseModel):
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

    filename: FilePath
    duration: Optional[NonNegativeFloat] = None
    fps: Optional[PositiveFloat] = None
    _fps_adjusted: Optional[PositiveFloat] = None
    video_annotation: Optional[VideoAnnotation] = None
    audio_annotation: Optional[SpeakerAnnotation] = None
    voice_features: Optional[VoiceFeatures] = None
    transcription: Optional[AudioTranscription] = None
    sentiment: Optional[SentimentAnnotation] = None
    features: Optional[pd.DataFrame] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    @computed_field
    def fps_adjusted(self) -> PositiveFloat:
        return self.fps if self._fps_adjusted is None else self._fps_adjusted

    @fps_adjusted.setter
    def fps_adjusted(self, value: PositiveFloat):
        self._fps_adjusted = value

    def _merge_video_annotation(self, data_frames: List[pd.DataFrame]):
        # create a new VideoAnnotation instance and copy all fields to the new instance
        # (except for average face embeddings) because the face embeddings have a different
        # dimension to the other fields in the instance. if we include the face embeddings
        # then the conversion of the data to a dataframe would fail.
        if self.video_annotation:
            video_annotation_dict = self.video_annotation.model_dump()
            del video_annotation_dict["face_average_embeddings"]
            data_frames.append(pd.DataFrame(video_annotation_dict))

    def _merge_audio_text_features(self, data_frames: List[pd.DataFrame]):
        if self.audio_annotation:
            audio_annotation_dict = {
                "frame": [],
                "segment_start": [],
                "segment_end": [],
                "segment_speaker_label": [],
            }

            time = np.arange(
                0.0, self.duration, 1 / self.fps_adjusted, dtype=np.float32
            )
            frame = np.arange(
                0,
                self.duration * self.fps,
                self.fps / self.fps_adjusted,
                dtype=np.int32,
            )

            if self.transcription:
                text_features_dict = {
                    "frame": [],
                    "span_start": [],
                    "span_end": [],
                    "span_text": [],
                    "segment_speaker_label": [],
                    "confidence": [],  # store confidence of transcription accuracy
                }

                if self.sentiment:
                    sentiment_dict = {
                        "frame": [],
                        "span_text": [],
                        "span_sent_pos": [],
                        "span_sent_neg": [],
                        "span_sent_neu": [],
                    }

            for i, t in zip(frame, time):
                overlap_segments = self.audio_annotation.segments[t]

                if len(overlap_segments) > 0:
                    for seg in overlap_segments:
                        audio_annotation_dict["frame"].append(i)
                        audio_annotation_dict["segment_start"].append(seg.begin)
                        audio_annotation_dict["segment_end"].append(seg.end)
                        audio_annotation_dict["segment_speaker_label"].append(
                            str(seg.data.name)
                        )
                else:
                    audio_annotation_dict["frame"].append(i)
                    audio_annotation_dict["segment_start"].append(None)
                    audio_annotation_dict["segment_end"].append(None)
                    audio_annotation_dict["segment_speaker_label"].append(None)

                if self.transcription:
                    for span in self.transcription.segments[t]:
                        text_features_dict["frame"].append(i)
                        text_features_dict["span_start"].append(span.begin)
                        text_features_dict["span_end"].append(span.end)
                        text_features_dict["span_text"].append(span.data.text)
                        text_features_dict["segment_speaker_label"].append(
                            span.data.speaker
                        )
                        text_features_dict["confidence"].append(
                            span.data.confidence
                        )  # store confidence of transcription accuracy

                    if self.sentiment:
                        for sent in self.sentiment.segments[t]:
                            sentiment_dict["frame"].append(i)
                            sentiment_dict["span_text"].append(sent.data.text)
                            sentiment_dict["span_sent_pos"].append(
                                sent.data.pos
                            )
                            sentiment_dict["span_sent_neg"].append(
                                sent.data.neg
                            )
                            sentiment_dict["span_sent_neu"].append(
                                sent.data.neu
                            )

            audio_text_features_df = pd.DataFrame(audio_annotation_dict)

            if self.transcription:
                text_features_df = pd.DataFrame(text_features_dict)
                if self.sentiment:
                    text_features_df = text_features_df.merge(
                        pd.DataFrame(sentiment_dict),
                        on=["frame", "span_text"],
                        how="left",
                    )

                audio_text_features_df = audio_text_features_df.merge(
                    text_features_df,
                    on=["frame", "segment_speaker_label"],
                    how="left",
                )

            data_frames.append(audio_text_features_df)

    def _merge_voice_features(self, data_frames: List):
        if self.voice_features:
            data_frames.append(pd.DataFrame(self.voice_features.model_dump()))

    @staticmethod
    def _delete_time_col(df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns:
            del df["time"]
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
            self.features = reduce(
                lambda left, right: pd.merge(
                    left, right, on=["frame"], how="left"
                ),
                dfs,
            )

            time = self.features.frame * (1 / self.fps)

            self.features.insert(1, "time", time)

        return self.features
