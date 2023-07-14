"""Download pretrained models when building Docker containter

This script downloads all standard pretrained models as part of building the Docker
container so they won't be downloaded every time the container is run.

"""

import argparse
import os


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--component", type=str, default="all")
    parser.add_argument("-t", "--token", type=str, default="")

    args = parser.parse_args()

    print("Downloading pretrained models...")

    if args.component == "vid":
        from mexca.video import FaceExtractor

        component = FaceExtractor(num_faces=2)
        component.detector
        component.encoder
        component.extractor

    elif args.component == "spe":
        from mexca.audio import SpeakerIdentifier

        component = SpeakerIdentifier(use_auth_token=args.token)
        component.pipeline

    elif args.component == "voi":
        from mexca.audio import VoiceExtractor

        component = VoiceExtractor()

    elif args.component == "tra":
        from mexca.text import AudioTranscriber

        component = AudioTranscriber()
        component.transcriber

    elif args.component == "sen":
        from mexca.text import SentimentExtractor

        component = SentimentExtractor()
        component.classifier

    else:
        raise Exception(
            "Please specify a valid component: 'vid', 'spe', 'voi', 'tra', 'sen'"
        )


if __name__ == "__main__":
    cli()
