"""Audio embedding — convert audio to dense vectors for semantic search."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sonarwise.core.models import AudioSegment

logger = logging.getLogger(__name__)


class BaseAudioEmbedder(ABC):
    """Base class for audio embedders. Implement embed_audio() and embed_text()."""

    @abstractmethod
    def embed_audio(self, audio: AudioSegment) -> np.ndarray:
        """Embed audio segment into a vector.

        Args:
            audio: AudioSegment to embed.

        Returns:
            1D numpy array (embedding_dim,).
        """
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text query into the same vector space as audio.

        Args:
            text: Text query to embed.

        Returns:
            1D numpy array (embedding_dim,).
        """
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors."""
        return 512


class CLAPEmbedder(BaseAudioEmbedder):
    """CLAP (Contrastive Language-Audio Pretraining) embedder.

    Joint text-audio embedding space: text queries match audio embeddings natively.

    Install: pip install sonarwise[clap]
    """

    def __init__(
        self,
        model: str = "laion/larger_clap_music_and_speech",
        device: Optional[str] = None,
    ):
        self.model_name = model
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            self._processor = ClapProcessor.from_pretrained(self.model_name)
            self._model = ClapModel.from_pretrained(self.model_name).to(device)
            self._model.eval()
            logger.info(f"CLAP model loaded: {self.model_name} on {device}")
        except ImportError:
            raise ImportError(
                "transformers/torch not installed. Run: pip install sonarwise[clap]"
            )

    def embed_audio(self, audio: AudioSegment) -> np.ndarray:
        self._load_model()
        import torch

        samples = audio.audio.astype(np.float32)
        sr = audio.sample_rate

        # CLAP requires 48kHz — resample if needed
        if sr != 48000:
            duration = len(samples) / sr
            target_len = int(duration * 48000)
            indices = np.linspace(0, len(samples) - 1, target_len).astype(int)
            samples = samples[indices]
            sr = 48000

        inputs = self._processor(
            audio=samples,
            sampling_rate=sr,
            return_tensors="pt",
        )

        # Move only tensor inputs to device
        audio_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
            if k in ("input_features", "is_longer")
        }

        with torch.no_grad():
            outputs = self._model.audio_model(**audio_inputs)
            # Get pooled output and project to shared space
            pooled = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
            if pooled is None:
                pooled = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.last_hidden_state.mean(dim=1)
            emb = self._model.audio_projection(pooled)

        emb = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_text(self, text: str) -> np.ndarray:
        self._load_model()
        import torch

        inputs = self._processor(
            text=text,
            return_tensors="pt",
        )

        # Move only tensor inputs to device
        text_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }

        with torch.no_grad():
            outputs = self._model.text_model(**text_inputs)
            pooled = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
            if pooled is None:
                pooled = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.last_hidden_state.mean(dim=1)
            emb = self._model.text_projection(pooled)

        emb = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @property
    def embedding_dim(self) -> int:
        return 512
