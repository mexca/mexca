"""Recipe for post-processing emotion expression features for further analysis.
"""

import argparse
import os

import polars as pl

from mexca.postprocessing import (
    AU_REF,
    LANDMARKS_REF,
    get_face_speaker_mapping,
    split_list_columns,
    sub_labels,
)


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")

    return parser.parse_args()


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
    feat_dfs = [
        sub_labels(df, mapping, column="face_label")
        for df, mapping in zip(feat_dfs, face_speaker_mappings)
    ]

    # Split list columns into separate columns
    feat_dfs = [
        split_list_columns(
            df, au_columns=AU_REF, landmark_columns=LANDMARKS_REF
        )
        for df in feat_dfs
    ]

    # Write to CSV files
    for i, filename in enumerate(feat_filenames):
        feat_dfs[i].write_csv(
            os.path.join(
                args.output_dir, os.path.splitext(filename)[0] + "_post.csv"
            )
        )


if __name__ == "__main__":
    main()
