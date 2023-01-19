"""Audio transcription and text sentiment extraction.
"""

import importlib.util


all = []

if importlib.util.find_spec('scipy') is not None:
    from .sentiment import SentimentExtractor
    all.append('SentimentExtractor')

if importlib.util.find_spec('whisper') is not None:
    from .transcription import AudioTranscriber
    all.append('AudioTranscriber')

__all__ = all
