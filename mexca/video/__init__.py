"""Facial feature extraction from video.
"""

import importlib.util

all = []

if importlib.util.find_spec("facenet_pytorch") is not None:
    from .extraction import FaceExtractor, VideoDataset

    all += ["FaceExtractor", "VideoDataset"]


__all__ = all
