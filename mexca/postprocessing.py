"""Post-process extracted emotion expression features."""

from itertools import product
from typing import Dict, Iterable, Union

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

AU_REF = [
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    32,
    38,
    39,
    "L1",
    "R1",
    "L2",
    "R2",
    "L4",
    "R4",
    "L6",
    "R6",
    "L10",
    "R10",
    "L12",
    "R12",
    "L14",
    "R14",
]
"""Names of facial action units."""


LANDMARKS_REF = list(range(1, 6))
"""Indices of facial landmarks."""


def split_list_columns(
    df: Union[pl.LazyFrame, pl.DataFrame],
    au_columns: Iterable,
    landmark_columns: Iterable,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Split (nested) list columns into separate columns.

    Parameters
    ----------
    df : polars.LazyFrame or polars.DataFrame
        Data frame with extracted emotion expression features as stored in :class:`Multimodal.features`.
    au_columns : Iterable
        Names for new facial action unit columns.
    landmark_columns : Iterable
        Names for new landmark columns.

    Notes
    -----
    For example, :class:`Pipeline.apply()` returns a `polars.LazyFrame`
    with (nested) list columns `face_box`, `face_au`, and `face_landmarks` containing multiple coordinates or predictions
    for multiple facial action units per row. These can be split into separate columns which only contain a single value per row.

    """
    df = (
        df.with_columns(
            pl.col("face_aus")
            .list.to_struct(upper_bound=len(au_columns))
            .struct.rename_fields(["face_au_" + str(au) for au in au_columns]),
            pl.col("face_box")
            .list.to_struct(upper_bound=4)
            .struct.rename_fields(
                ["face_box_x1", "face_box_y1", "face_box_x2", "face_box_y2"]
            ),
            pl.col("face_landmarks")
            .list.to_struct(upper_bound=len(landmark_columns))
            .struct.rename_fields(
                ["face_landmarks_" + str(i) for i in landmark_columns]
            ),
        )
        .unnest(columns=["face_box", "face_aus", "face_landmarks"])
        .with_columns(
            [
                pl.col("face_landmarks_" + str(i))
                .list.to_struct(upper_bound=2)
                .struct.rename_fields(
                    ["face_landmarks_x" + str(i), "face_landmarks_y" + str(i)]
                )
                for i in landmark_columns
            ]
        )
        .unnest(columns=["face_landmarks_" + str(i) for i in landmark_columns])
    )

    return df


def get_face_speaker_mapping(
    df: pl.DataFrame,
    face_label_column_name: str = "face_label",
    speaker_label_column_name: str = "segment_speaker_label",
) -> Dict[str, str]:
    """Get optimal mapping between face and speaker labels by counting overlapping frames.

    Uses the Hungarian algorithm to find an optimal mapping between face and speaker labels.

    Parameters
    ----------
    df : polars.DataFrame
        Data frame with columns `face_label_column_name` and speaker_label_column_name`.
    face_label_column_name : str, default="face_label"
        Name of the face label column.
    speaker_label_column_name : str, default="segment_speaker_label"
        Name of the speaker label column.

    """
    df = df.drop_nulls([face_label_column_name, speaker_label_column_name])

    face_labels = df[face_label_column_name].to_numpy()
    speaker_labels = df[speaker_label_column_name].to_numpy()

    x_labels = np.unique(face_labels)
    y_labels = np.unique(speaker_labels)

    # Init cost matrix
    cost_mat = np.zeros((len(x_labels), len(y_labels)))

    for x, y in zip(
        face_labels,
        speaker_labels,
    ):
        # Get unique detected faces (some faces are duplicates for different speakers) larger than minimum height
        x = np.unique(np.array(x))

        # Loop through pairs
        for match in product(x, y):
            # If pair elements match increase cost matrix cell
            cost_mat[
                np.where(x_labels == match[0]), np.where(y_labels == match[1])
            ] += 1

    # Get mapping from cost matrix
    rows, cols = linear_sum_assignment(-cost_mat, maximize=False)

    mapping = {str(int(r)): str(int(y_labels[c])) for r, c in zip(rows, cols)}

    return mapping


def sub_labels(
    df: Union[pl.LazyFrame, pl.DataFrame], mapping: Dict, column: str
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Replace label column with labels from a mapping.

    df : polars.LazyFrame or polars.DataFrame
        Data frame with label column.
    mapping : dict
        Dictionary with mapping.
    column : str
        Name of label column in data frame.

    """
    df = df.with_columns(pl.col(column).map_dict(mapping, default="-1"))
    return df
