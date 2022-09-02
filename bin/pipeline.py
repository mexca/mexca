"""Apply the mexca pipeline to a video file from the command line.

This script is used by the docker container.

"""

import argparse
import json
from json import JSONEncoder
import numpy as np
from mexca.audio.extraction import VoiceExtractor
from mexca.audio.identification import SpeakerIdentifier
from mexca.audio.integration import AudioIntegrator
from mexca.core.pipeline import Pipeline
from mexca.text.transcription import AudioTextIntegrator
from mexca.text.transcription import AudioTranscriber
from mexca.video.extraction import FaceExtractor

class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.ndarray, np.float32, np.chararray)):
            return o.tolist()
        return JSONEncoder.default(self, o)


parser = argparse.ArgumentParser(description='Apply the mexca pipeline to a video file.')
parser.add_argument('-f', '--filepath', type=str, required=True, dest='filepath')
parser.add_argument('-o', '--output', type=str, required=True, dest='output')
parser.add_argument('-n', '--n-clusters', type=int, default=2, dest='n_clusters')
parser.add_argument('-l', '--lang', type=str, default='english', dest='language')
parser.add_argument('--skip', type=int, default=1, dest='skip_frames')
parser.add_argument('--subclip', nargs=2, default=[0, None], dest='process_subclip')
parser.add_argument('--no-video', action='store_false', dest='no_video')
parser.add_argument('--no-audio', action='store_false', dest='no_audio')
parser.add_argument('--no-text', action='store_false', dest='no_text')

args = parser.parse_args()

pipeline = Pipeline(
    video=None if args.no_video else FaceExtractor(
        num_clusters=args.n_clusters
    ),
    audio=None if args.no_audio else AudioIntegrator(
        SpeakerIdentifier(num_speakers=args.n_clusters),
        VoiceExtractor()
    ),
    text=None if args.no_text else AudioTextIntegrator(
        audio_transcriber=AudioTranscriber(args.language)
    )
)

output = pipeline.apply(
    filepath=args.filepath,
    skip_frames=args.skip_frames,
    process_subclip=args.process_subclip
)

with open(args.output, 'w', encoding='utf-8') as file:
    json.dump(output.features, file, cls=NumpyArrayEncoder, indent=2)
