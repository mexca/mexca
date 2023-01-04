"""Voice feature extraction from video audio.
"""

from .identification import SpeakerIdentifier
from .extraction import VoiceExtractor

__all__ = ['SpeakerIdentifier', 'VoiceExtractor']
