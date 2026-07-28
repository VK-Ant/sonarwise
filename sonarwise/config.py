"""Configuration system for sonarwise."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class SonarWiseConfig:
    """Global configuration for sonarwise."""

    # Models
    transcriber_model: str = "base"
    embedder_model: str = "laion/larger_clap_music_and_speech"
    diarizer_model: str = "pyannote/speaker-diarization-3.1"
    event_model: str = "CNN14"
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # Hardware
    device: Optional[str] = None  # auto-detect
    compute_type: str = "float16"

    # Features
    diarization: bool = False
    events: bool = False
    word_timestamps: bool = True

    # Storage
    db_path: str = "sonarwise.db"

    # Chunking
    min_segment_ms: int = 500
    max_segment_ms: int = 30000
    silence_threshold_ms: int = 300
    padding_ms: int = 200

    # Search
    default_top_k: int = 5
    similarity_threshold: float = 0.0

    # Live
    chunk_duration_ms: int = 2000
    sample_rate: int = 16000

    # Logging
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SonarWiseConfig":
        """Load config from YAML file."""
        if not os.path.exists(path):
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Flatten nested config
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}_{subkey}"] = subvalue
            else:
                flat[key] = value

        for key, value in flat.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        data = {
            "transcriber": {"model": self.transcriber_model},
            "embedder": {"model": self.embedder_model},
            "diarizer": {"model": self.diarizer_model},
            "event_classifier": {
                "model": self.event_model,
            },
            "device": self.device,
            "compute_type": self.compute_type,
            "diarization": self.diarization,
            "events": self.events,
            "db_path": self.db_path,
            "chunker": {
                "min_segment_ms": self.min_segment_ms,
                "max_segment_ms": self.max_segment_ms,
            },
            "search": {
                "default_top_k": self.default_top_k,
                "similarity_threshold": self.similarity_threshold,
            },
            "live": {
                "chunk_duration_ms": self.chunk_duration_ms,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def auto_load(cls) -> "SonarWiseConfig":
        """Auto-load config from sonarwise.yaml if it exists."""
        candidates = [
            "sonarwise.yaml",
            "sonarwise.yml",
            os.path.expanduser("~/.sonarwise.yaml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return cls.from_yaml(path)
        return cls()
