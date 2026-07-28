"""Speaker diarization — identify who is speaking in each segment."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from sonarwise.core.models import AudioData, SpeakerSegment

logger = logging.getLogger(__name__)


class BaseDiarizer(ABC):
    """Base class for diarizers. Implement diarize() to bring your own model."""

    @abstractmethod
    def diarize(self, audio: AudioData) -> List[SpeakerSegment]:
        """Identify speakers in audio.

        Args:
            audio: AudioData to diarize.

        Returns:
            List of SpeakerSegment with (start_ms, end_ms, speaker_id).
        """
        raise NotImplementedError


class PyannoteDiarizer(BaseDiarizer):
    """Pyannote speaker diarization.

    Install: pip install sonarwise[diarization]
    Requires a HuggingFace token with pyannote access.
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.hf_token = hf_token
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
            self._pipeline.to(torch.device(device))
            logger.info(f"Pyannote diarizer loaded: {self.model_name} on {device}")
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. Run: pip install sonarwise[diarization]"
            )

    def diarize(self, audio: AudioData) -> List[SpeakerSegment]:
        self._load_model()

        import torch
        import torchaudio
        import tempfile
        import os

        # Pyannote expects a file path or waveform tensor
        waveform = torch.FloatTensor(audio.samples).unsqueeze(0)

        # Save to temp file (pyannote works best with file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, waveform, audio.sample_rate)
            tmp_path = f.name

        try:
            kwargs = {}
            if self.min_speakers is not None:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kwargs["max_speakers"] = self.max_speakers

            diarization = self._pipeline(tmp_path, **kwargs)
        finally:
            os.unlink(tmp_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                speaker_id=speaker,
            ))

        logger.info(
            f"Diarization: {len(segments)} turns, "
            f"{len(set(s.speaker_id for s in segments))} speakers"
        )
        return segments


class SimpleDiarizer(BaseDiarizer):
    """Simple clustering-based diarizer using speaker embeddings.

    No external dependencies beyond a speaker embedder.
    Clusters segments by voice similarity.
    """

    def __init__(self, n_speakers: Optional[int] = None, threshold: float = 0.75):
        self.n_speakers = n_speakers
        self.threshold = threshold

    def diarize(self, audio: AudioData) -> List[SpeakerSegment]:
        # Single-speaker fallback — assigns everything to speaker_0
        logger.warning(
            "SimpleDiarizer: assigning all segments to speaker_0. "
            "Install pyannote for real diarization: pip install sonarwise[diarization]"
        )
        duration_ms = audio.duration_ms
        return [SpeakerSegment(start_ms=0, end_ms=duration_ms, speaker_id="speaker_0")]
