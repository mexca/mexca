"""Recipe for applying the standard MEXCA pipeline to a series of videos
and write the output to JSON files.
"""

import argparse
import logging
import os

import numpy as np
import torch
import yaml

from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.pipeline import Pipeline
from mexca.text import AudioTranscriber, SentimentExtractor
from mexca.utils import optional_float
from mexca.video import FaceExtractor


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-faces", type=int, default=2)
    parser.add_argument("--num-speakers", type=int, default=2)
    # Use 'tiny' for testing and 'large' for full run
    parser.add_argument("--whisper-model", type=str, default="tiny")
    # Set according to available RAM
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--skip-frames", type=int, default=5)
    # Very useful for testing!
    parser.add_argument(
        "--process-subclip", type=optional_float, nargs=2, default=[0, None]
    )

    return parser.parse_args()


def main():
    args = cli()

    # Get a filenames of MP4 files in input dir
    filenames = os.listdir(args.input_dir)
    filenames = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in filenames
            if f.endswith(".mp4")
        ]
    )

    # Run with CUDA if available
    device = (
        torch.device(type="cuda")
        if torch.cuda.is_available()
        else torch.device(type="cpu")
    )

    # Create pipeline
    pipeline = Pipeline(
        face_extractor=FaceExtractor(num_faces=args.num_faces, device=device),
        speaker_identifier=SpeakerIdentifier(
            num_speakers=args.num_speakers,
            # Requires Hugging Face authentication
            use_auth_token=os.environ["HF_TOKEN"]
            if "HF_TOKEN" in os.environ
            else True,
        ),
        voice_extractor=VoiceExtractor(),
        audio_transcriber=AudioTranscriber(
            whisper_model=args.whisper_model, device=device
        ),
        sentiment_extractor=SentimentExtractor(device=device),
    )

    # Load logging configuration
    with open(os.path.join("logging.yml"), "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    try:
        pipeline.logger.info("Processing %s ...", filenames)

        # Set seeds
        np.random.seed(1)
        torch.manual_seed(1)

        # Apply pipeline
        results = pipeline.apply(
            filepath=filenames,
            frame_batch_size=args.batch_size,
            skip_frames=args.skip_frames,
            process_subclip=args.process_subclip,
        )

        # Save results for each file
        for filename, result in zip(filenames, results):
            # Remove path and only keep filename without extension
            base_name = os.path.splitext(os.path.split(filename)[-1])[0]

            # Trigger lazy evaluation
            features_df = result.features.collect()

            features_df.write_json(
                os.path.join(
                    args.output_dir, f"mexca_{base_name}_features.json"
                )
            )
            result.video_annotation.write_json(
                os.path.join(
                    args.output_dir, f"mexca_{base_name}_video_annotation.json"
                )
            )
            result.audio_annotation.write_json(
                os.path.join(
                    args.output_dir,
                    f"mexca_{base_name}_speaker_annotation.json",
                )
            )
            result.voice_features.write_json(
                os.path.join(
                    args.output_dir, f"mexca_{base_name}_voice_features.json"
                )
            )
            result.transcription.write_json(
                os.path.join(
                    args.output_dir, f"mexca_{base_name}_transcription.json"
                )
            )
            result.sentiment.write_json(
                os.path.join(
                    args.output_dir, f"mexca_{base_name}_sentiment.json"
                )
            )

    # Log exceptions
    except Exception as exc:
        pipeline.logger.error("%s", exc, exc_info=True)
        raise exc


if __name__ == "__main__":
    main()
