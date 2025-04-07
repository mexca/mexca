"""Recipe for annotating a video with features.

Specifically:
- Annotates face bounding boxes and labels
- Annotates current speaker labels and speech transcriptions
- Adds three plots with emotion expression features over time:
    - AU12 activation
    - Voice pitch
    - Positive speech sentiment

Requires opencv, polars, and seaborn to be installed.

"""

import argparse
from collections import deque

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set plot theme
sns.set_theme()
sns.set_style("white")
sns.set_style("ticks")

# Map candidate labels to colors
CANDIDATE_COLORS = {
    key: val for key, val in zip(range(-1, 9), sns.color_palette())
}

# Transform colors from RGB to BGR (for openCV)
CANDIDATE_COLORS_BGR = {
    key: [c * 255 for c in reversed(colors.to_rgb(CANDIDATE_COLORS[key]))]
    for key in CANDIDATE_COLORS
}

# Define font properties
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.65
FONT_LINE_TYPE = cv2.LINE_AA


def draw_face_boxes(frame, features):
    """Draw face boxes and labels on a video frame."""
    for row in features.iter_rows(named=True):
        if not np.isnan(row["face_prob"]):
            x1 = int(row["face_box_x1"])
            x2 = int(row["face_box_x2"])
            y1 = int(row["face_box_y1"])
            y2 = int(row["face_box_y2"])

            lbl = row["face_label"]
            # Draw face box rectangle
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), CANDIDATE_COLORS_BGR[lbl], 2
            )

            x3 = int(x1 + (x2 - x1) / 2)
            y3 = y1 - 5

            face_label = str(lbl)

            # Add face labels
            face_label_size = cv2.getTextSize(face_label, FONT, FONT_SCALE, 1)[
                0
            ]

            cv2.putText(
                frame,
                face_label,
                (x3 - int(face_label_size[0] / 2), y3),
                FONT,
                FONT_SCALE,
                CANDIDATE_COLORS_BGR[lbl],
                1,
                lineType=FONT_LINE_TYPE,
            )


def draw_speaker_labels(frame, features):
    """Draw speaker labels on a video frame."""
    # span_texts = features.select(pl.col("span_text")).to_series()

    txt_pos = (50, 50)

    for j, spk in enumerate(features["segment_speaker_label"].unique()):
        if spk is not None:
            speaker_label = str(spk)
            speaker_label_size = cv2.getTextSize(
                speaker_label, FONT, FONT_SCALE, 1
            )[0]
            speaker_label_pos = (
                50,
                txt_pos[1] + j * speaker_label_size[1] + speaker_label_size[1],
            )
            cv2.putText(
                frame,
                speaker_label,
                speaker_label_pos,
                FONT,
                FONT_SCALE,
                CANDIDATE_COLORS_BGR[spk],
                1,
                lineType=FONT_LINE_TYPE,
            )

            span_texts = features.filter(
                pl.col("segment_speaker_label").eq(spk)
            )["span_text"].unique()

            for span_text in span_texts:
                if span_text is not None:
                    span_text_split = span_text.split()
                    txts = []

                    while len(span_text_split) > 0:
                        line = ""
                        while len(line) < 30 and len(span_text_split) > 0:
                            line += span_text_split.pop(0) + " "
                        txts.append(line[:-1])

                    for k, txt in enumerate(txts):
                        if txt:
                            txt_size = cv2.getTextSize(
                                txt, FONT, FONT_SCALE, 1
                            )[0]
                            txt_pos = (
                                50,
                                speaker_label_pos[1] + (k + 1) * txt_size[1],
                            )
                            cv2.putText(
                                frame,
                                txt,
                                txt_pos,
                                FONT,
                                FONT_SCALE,
                                CANDIDATE_COLORS_BGR[spk],
                                1,
                                lineType=FONT_LINE_TYPE,
                            )


def create_line_plot(time, label, y, width, height):
    """Create plot for lines per label over time."""
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(
        1, 1, figsize=(width * px, height * px), constrained_layout=True
    )

    # Convert to arrays
    time = np.array(list(time))
    label = np.array(list(label))
    y = np.array(list(y))

    # Plot y over time for each label
    for lbl in np.unique(label):
        # Omit nan labels
        if np.isfinite(lbl):
            mask = label == lbl
            y_sub = y[mask]
            time_sub = time[mask]

            ax.plot(time_sub, y_sub, color=CANDIDATE_COLORS[lbl])

    # Adjust time limits
    lower_lim = time.min() if time.size > 0 else 0
    upper_lim = time.max() if time.size > 0 else 0
    ax.set_xlim(lower_lim, upper_lim)

    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    plt.close()
    return fig, ax


def fig_to_img(fig):
    """Convert a maplotlib figure to a image array (for openCV)."""
    # fig.tight_layout()
    fig.canvas.draw()
    fig_array = fig.canvas.buffer_rgba()
    return np.asarray(fig_array)[:, :, (2, 1, 0)]


def cli(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--video-filename", help="Path to the original video file."
    )
    parser.add_argument(
        "--results-filename",
        help="Path to the results file corresponding to the original video file.",
    )
    parser.add_argument(
        "--output-filename", help="Path to the annotated output video file."
    )
    parser.add_argument(
        "--starttime",
        default=0,
        type=int,
        help="Start time of annotation (in seconds).",
    )
    parser.add_argument(
        "--endtime",
        default=10,
        type=int,
        help="End time of annotation (in seconds).",
    )

    return parser.parse_args(args)


def main(args):
    """Annotate video."""
    args = cli(args)

    # Get postprocessed feature df from video file
    feature_df = (
        pl.scan_csv(
            args.results_filename  # CHANGE INPUT FEATURE FILENAME HERE
        ).with_columns(
            pl.col("face_label").cast(pl.Int32).alias("face_label"),
            pl.col("segment_speaker_label")
            .cast(pl.Int32)
            .alias("segment_speaker_label"),
            pl.col("face_au_12").cast(pl.Float64).alias("face_au_12"),
            pl.col("pitch_f0_hz").cast(pl.Float64).alias("pitch_f0_hz"),
            pl.col("span_sent_pos").cast(pl.Float64).alias("span_sent_pos"),
        )
    ).collect()

    # Define start and end of annotated video
    t_start = args.starttime  # CHANGE START AND END TIME HERE
    t_end = args.endtime

    # Open video stream
    cap = cv2.VideoCapture(
        args.video_filename
    )  # CHANGE INPUT VIDEO FILENAME HERE

    # Get size for saving
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Adapt wiwdth to make space for feature plots
    size = (int(width * (4 / 3)), height)

    # Open video writer
    annotated = cv2.VideoWriter(
        args.output_filename,  # CHANGE TARGET VIDEO FILENAME HERE
        cv2.VideoWriter_fourcc(*"MJPG"),
        25,
        size,
    )

    # Number of frames to show at same time in feature plots
    num_frames = 100

    # Init couners
    i = 0
    t = 0.0

    # Helper variable
    valid_row = None

    time = deque(maxlen=num_frames)
    face_label = deque(maxlen=num_frames)
    segment_speaker_label = deque(maxlen=num_frames)
    face_au = deque(maxlen=num_frames)
    pitch = deque(maxlen=num_frames)
    sent = deque(maxlen=num_frames)

    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        # Stop if no input received
        if not ret:
            break

        if t > t_start and t < t_end:
            # Get matching outut from pipeline
            frame_row = feature_df.filter(pl.col("frame") == i)

            # Only update if there is a matching frame in the feauture df,
            # otherwise keep old features
            if frame_row.shape[0] > 0 or valid_row is None:
                # Update old valid features
                valid_row = frame_row

                time.extend(frame_row["time"].to_numpy().tolist())
                face_label.extend(frame_row["face_label"].to_numpy().tolist())
                segment_speaker_label.extend(
                    frame_row["segment_speaker_label"].to_numpy().tolist()
                )
                face_au.extend(frame_row["face_au_12"].to_numpy().tolist())
                pitch.extend(frame_row["pitch_f0_hz"].to_numpy().tolist())
                sent.extend(frame_row["span_sent_pos"].to_numpy().tolist())

                # Create feature plots
                au_plot, ax = create_line_plot(
                    time,
                    face_label,
                    face_au,
                    width=width / 3,
                    height=height / 3,
                )

                ax.set_yticks(np.arange(0, 1.2, step=0.2))
                ax.set_title("Lip corner puller (AU12) activation", loc="left")

                au_img = fig_to_img(au_plot)

                pitch_plot, ax = create_line_plot(
                    time,
                    segment_speaker_label,
                    pitch,
                    width=width / 3,
                    height=height / 3,
                )

                ax.set_ylim((0, 400))
                ax.set_title("Voice pitch (F0 in Hz)", loc="left")

                pitch_img = fig_to_img(pitch_plot)

                sent_plot, ax = create_line_plot(
                    time,
                    segment_speaker_label,
                    sent,
                    width=width / 3,
                    height=height / 3,
                )

                ax.set_yticks(np.arange(0, 1.2, step=0.2))
                ax.set_title("Positive speech sentiment", loc="left")

                sent_img = fig_to_img(sent_plot)

            draw_face_boxes(frame, valid_row)

            draw_speaker_labels(frame, valid_row)

            # Concatenate annotated video frame with feature plots
            frame_concat = cv2.hconcat(
                [
                    frame,
                    cv2.resize(
                        cv2.vconcat([au_img, pitch_img, sent_img]),
                        (int(width / 3), int(height)),
                        interpolation=cv2.INTER_AREA,
                    ),
                ]
            )

            annotated.write(frame_concat)

        if t > t_end:
            break

        i += 1
        t += 0.04
        print(i, t)

    cap.release()
    annotated.release()


if __name__ == "__main__":
    main(args=None)
