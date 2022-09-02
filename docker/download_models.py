"""Download pretrained models when building Docker containter

This script downloads all standard pretrained models as part of building the Docker
container so they won't be downloaded every time the container is run.

"""

from mexca.core.pipeline import Pipeline

# Trigger model download by creating Pipeline instance
print('Download pretrained models...')
pipeline = Pipeline().from_default()
