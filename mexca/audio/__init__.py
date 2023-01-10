"""Voice feature extraction from video audio.
"""

import importlib.util

all = []

if importlib.util.find_spec('pyannote.audio') is not None:
    from .identification import SpeakerIdentifier
    all.append('SpeakerIdentifier')
if importlib.util.find_spec('praat-parselmouth') is not None:
    from .extraction import VoiceExtractor
    all.append('VoiceExtractor')

__all__ = all
