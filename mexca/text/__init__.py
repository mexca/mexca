"""Audio transcription and text sentiment extraction.
"""

import importlib.util
from .sentiment import SentimentExtractor

all = ['SentimentExtractor']

if importlib.util.find_spec('whisper') is not None:
    from .transcription import AudioTranscriber
    all.append('AudioTranscriber')

__all__ = all
