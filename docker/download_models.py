"""Download pretrained models when building Docker containter

This script downloads all standard pretrained models as part of building the Docker
container so they won't be downloaded every time the container is run.

"""

import argparse

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--component', type=str, default='all')

    args = parser.parse_args()

    print('Downloading pretrained models...')

    if args.component == 'vid':
        from mexca.video import FaceExtractor

        component = FaceExtractor(num_faces=2)

    elif args.component == 'spe':
        from mexca.audio import SpeakerIdentifier

        component = SpeakerIdentifier()

    elif args.component == 'voi':
        pass

    elif args.component == 'tra':
        from mexca.text import AudioTranscriber

        component = AudioTranscriber

    elif args.component == 'sen':
        from mexca.text import SentimentExtractor

        component = SentimentExtractor

    else:
        raise Exception("Please specify a valid component: 'vid', 'spe', 'voi', 'tra', 'sen'")


if __name__ == '__main__':
    cli()
