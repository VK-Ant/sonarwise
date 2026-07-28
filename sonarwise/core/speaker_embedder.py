"""Speaker embedding — generate voice fingerprints for identification."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sonarwise.core.models import AudioSegment
from sonarwise.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class BaseSpeakerEmbedder(ABC):
    """Base class for speaker embedders. Implement embed() to bring your own model."""

    @abstractmethod
    def embed(self, audio: AudioSegment) -> np.ndarray:
        """Generate speaker voiceprint from audio.

        Args:
            audio: AudioSegment of one speaker.

        Returns:
            1D numpy array (embedding_dim,).
        """
        raise NotImplementedError

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compare two speaker embeddings. Returns similarity score [0, 1]."""
        return cosine_similarity(emb_a, emb_b)

    @property
    def embedding_dim(self) -> int:
        return 192


class ECAPAEmbedder(BaseSpeakerEmbedder):
    """ECAPA-TDNN speaker embedder via SpeechBrain.

    Install: pip install sonarwise[speaker]
    """

    def __init__(
        self,
        model: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
    ):
        self.model_name = model
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                run_opts={"device": device},
            )
            logger.info(f"ECAPA embedder loaded: {self.model_name} on {device}")
        except ImportError:
            raise ImportError(
                "speechbrain not installed. Run: pip install sonarwise[speaker]"
            )

    def embed(self, audio: AudioSegment) -> np.ndarray:
        self._load_model()
        import torch

        waveform = torch.FloatTensor(audio.audio).unsqueeze(0)
        with torch.no_grad():
            embedding = self._model.encode_batch(waveform)

        emb = embedding.squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @property
    def embedding_dim(self) -> int:
        return 192
