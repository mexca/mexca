"""Recipe for post-processing emotion expression features for further analysis.
"""

import argparse
import os
from itertools import product
from typing import Dict, Union

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


LMK_REF = list(range(1, 6))
"""Indices of facial landmarks."""


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")

    return parser.parse_args()


def split_list_columns(
    df: Union[pl.LazyFrame, pl.DataFrame]
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Split (nested) list columns into separate columns."""
    df = (
        df.with_columns(
            pl.col("face_aus")
            .list.to_struct()
            .struct.rename_fields(["face_au_" + str(au) for au in AU_REF]),
            pl.col("face_box")
            .list.to_struct()
            .struct.rename_fields(
                ["face_box_x1", "face_box_y1", "face_box_x2", "face_box_y2"]
            ),
            pl.col("face_landmarks")
            .list.to_struct()
            .struct.rename_fields(
                ["face_landmarks_" + str(i) for i in LMK_REF]
            ),
        )
        .unnest(columns=["face_box", "face_aus", "face_landmarks"])
        .with_columns(
            [
                pl.col("face_landmarks_" + str(i))
                .list.to_struct()
                .struct.rename_fields(
                    ["face_landmarks_x" + str(i), "face_landmarks_y" + str(i)]
                )
                for i in LMK_REF
            ]
        )
        .unnest(columns=["face_landmarks_" + str(i) for i in LMK_REF])
    )

    return df


def get_face_speaker_mapping(df: pl.DataFrame) -> pl.DataFrame:
    """Get optimal mapping between face and speaker labels by counting overlapping frames."""
    df = df.drop_nulls(["face_label", "segment_speaker_label"])

    x_labels = df.select(pl.col("face_label").unique()).to_numpy()
    y_labels = df.select(pl.col("segment_speaker_label").unique()).to_numpy()

    # Init cost matrix
    cost_mat = np.zeros((len(x_labels), len(y_labels)))

    for x, y in zip(
        df.select(pl.col("face_label")).to_series(),
        df.select(pl.col("segment_speaker_label")).to_series(),
    ):
        # Get unique detected faces (some faces are duplicates for different speakers) larger than minimum height
        x = np.unique(np.array(x))

        # Create nested loop pairs
        matches = product(x, y)

        # Loop through pairs
        for match in matches:
            # If pair elements match increase cost matrix cell
            cost_mat[
                np.where(x_labels == match[0]), np.where(y_labels == match[1])
            ] += 1

    # Get mapping from cost matrix
    rows, cols = linear_sum_assignment(-cost_mat, maximize=False)

    mapping = {}

    # Assign labels to mapping
    for r, c in zip(rows, cols):
        mapping[str(int(r))] = str(int(y_labels[c]))

    return mapping


def sub_face_labels(
    df: Union[pl.LazyFrame, pl.DataFrame], mapping: Dict
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Replace face labels with labels from a mapping."""
    df = df.with_columns(pl.col("face_label").map_dict(mapping, default="-1"))
    return df


def main():
    args = cli()

    # Get filenames of feature JSON files
    feat_filenames = sorted(
        [
            f"mexca_{os.path.splitext(os.path.split(filename)[-1])[0]}_features.json"
            for filename in os.listdir(args.input_dir)
            if filename.endswith(".mp4")
        ]
    )

    # Load JSON files
    feat_dfs = pl.Series(
        [
            pl.read_json(os.path.join(args.output_dir, filename))
            for filename in feat_filenames
        ]
    )

    # Get mappings between face and speaker labels
    face_speaker_mappings = [get_face_speaker_mapping(df) for df in feat_dfs]

    # Replace face labels by mapped speaker labels
    feat_dfs = pl.Series(
        [
            sub_face_labels(df, mapping)
            for df, mapping in zip(feat_dfs, face_speaker_mappings)
        ]
    )

    # Split list columns into separate columns
    feat_dfs = feat_dfs.map_elements(split_list_columns)

    # Write to CSV files
    for i, filename in enumerate(feat_filenames):
        feat_dfs[i].write_csv(
            os.path.join(args.output_dir, filename.split(".")[0] + "_post.csv")
        )


if __name__ == "__main__":
    main()
