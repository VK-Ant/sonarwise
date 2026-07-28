"""
sonarwise - Pluggable audio perception engine.
Hear. Search. Retrieve.

Usage:
    from sonarwise import SonarWise

    sw = SonarWise()
    sw.index("meeting.wav")
    results = sw.query("budget discussion", top_k=5)
"""

__version__ = "0.1.0"
__author__ = "VK-Ant (Venkatkumar Rajan)"
__tagline__ = "Hear. Search. Retrieve."

from sonarwise.core.pipeline import SonarWise
from sonarwise.core.models import (
    AudioData,
    AudioSegment,
    SearchResult,
    EventResult,
    EventTag,
    Speaker,
    SpeakerStats,
    TranscriptResult,
    Word,
    SpeakerSegment,
    Source,
)

__all__ = [
    "SonarWise",
    "AudioData",
    "AudioSegment",
    "SearchResult",
    "EventResult",
    "EventTag",
    "Speaker",
    "SpeakerStats",
    "TranscriptResult",
    "Word",
    "SpeakerSegment",
    "Source",
]
