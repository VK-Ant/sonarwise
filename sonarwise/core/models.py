"""Data models for sonarwise."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AudioData:
    """Raw audio data in standard format (16kHz mono float32)."""

    samples: "np.ndarray"  # noqa: F821
    sample_rate: int = 16000
    duration_ms: int = 0
    channels: int = 1
    source_path: Optional[str] = None

    def __post_init__(self):
        if self.duration_ms == 0 and self.sample_rate > 0:
            self.duration_ms = int(len(self.samples) / self.sample_rate * 1000)


@dataclass
class AudioInfo:
    """Metadata about an audio file without loading it."""

    duration_ms: int
    sample_rate: int
    channels: int
    format: str
    size_mb: float


@dataclass
class AudioSegment:
    """A chunk of audio with timing info."""

    audio: "np.ndarray"  # noqa: F821
    sample_rate: int
    start_ms: int
    end_ms: int
    duration_ms: int = 0

    def __post_init__(self):
        if self.duration_ms == 0:
            self.duration_ms = self.end_ms - self.start_ms


@dataclass
class Word:
    """Single word with timestamp."""

    text: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptResult:
    """Transcription output for a segment."""

    text: str
    language: str = "en"
    confidence: float = 0.0
    words: List[Word] = field(default_factory=list)


@dataclass
class EventTag:
    """Detected audio event."""

    label: str
    confidence: float


@dataclass
class SpeakerSegment:
    """Speaker diarization output for a time range."""

    start_ms: int
    end_ms: int
    speaker_id: str


@dataclass
class Segment:
    """Complete indexed segment — the atomic unit stored in the vector store."""

    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    filepath: str = ""
    start_ms: int = 0
    end_ms: int = 0
    duration_ms: int = 0
    transcript: str = ""
    embedding: Optional[List[float]] = None
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    speaker_embedding: Optional[List[float]] = None
    event_tags: List[str] = field(default_factory=list)
    event_confidence: List[float] = field(default_factory=list)
    language: str = "en"
    word_timestamps: List[Word] = field(default_factory=list)
    source_type: str = "batch"
    indexed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        if self.duration_ms == 0:
            self.duration_ms = self.end_ms - self.start_ms


@dataclass
class SearchResult:
    """Query result returned to the user."""

    segment_id: str = ""
    filepath: str = ""
    start_ms: int = 0
    end_ms: int = 0
    duration_ms: int = 0
    transcript: str = ""
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    event_tags: List[str] = field(default_factory=list)
    event_confidence: List[float] = field(default_factory=list)
    language: str = "en"
    word_timestamps: List[Word] = field(default_factory=list)
    score: float = 0.0
    source_type: str = "batch"


@dataclass
class EventResult:
    """Event-only query result."""

    filepath: str = ""
    start_ms: int = 0
    end_ms: int = 0
    event_tags: List[str] = field(default_factory=list)
    event_confidence: List[float] = field(default_factory=list)
    transcript: str = "[non-speech]"
    score: float = 0.0


@dataclass
class Speaker:
    """Speaker info within a file."""

    id: str
    name: Optional[str] = None
    duration_ms: int = 0
    segment_count: int = 0


@dataclass
class SpeakerStats:
    """Detailed speaker statistics."""

    speaker_id: str
    speaker_name: Optional[str] = None
    total_duration_ms: int = 0
    percentage: float = 0.0
    segment_count: int = 0
    avg_segment_ms: float = 0.0
    longest_segment_ms: int = 0
    word_count: int = 0


@dataclass
class RegisteredSpeaker:
    """A registered speaker with voiceprint."""

    name: str
    embedding: List[float]
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Source:
    """Indexed audio source info."""

    filepath: str
    segment_count: int = 0
    duration_ms: int = 0
    indexed_at: str = ""
    speakers: int = 0
