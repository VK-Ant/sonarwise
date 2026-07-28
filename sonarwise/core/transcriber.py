"""Speech transcription — convert audio segments to text."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sonarwise.core.models import AudioSegment, TranscriptResult, Word

logger = logging.getLogger(__name__)


class BaseTranscriber(ABC):
    """Base class for transcribers. Implement transcribe() to bring your own model."""

    @abstractmethod
    def transcribe(self, audio: AudioSegment) -> TranscriptResult:
        """Transcribe an audio segment to text.

        Args:
            audio: AudioSegment to transcribe.

        Returns:
            TranscriptResult with text, language, confidence, and word timestamps.
        """
        raise NotImplementedError


class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper transcriber.

    Install: pip install sonarwise[whisper]
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        word_timestamps: bool = True,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.word_timestamps = word_timestamps
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            device = self.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = whisper.load_model(self.model_size, device=device)
            logger.info(f"Whisper {self.model_size} loaded on {device}")
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install sonarwise[whisper]"
            )

    def transcribe(self, audio: AudioSegment) -> TranscriptResult:
        self._load_model()

        samples = audio.audio.astype(np.float32)

        options = {"language": self.language}
        if self.word_timestamps:
            options["word_timestamps"] = True

        result = self._model.transcribe(samples, **options)

        words = []
        if self.word_timestamps and "segments" in result:
            for seg in result["segments"]:
                for w in seg.get("words", []):
                    words.append(Word(
                        text=w["word"].strip(),
                        start_ms=int(w["start"] * 1000),
                        end_ms=int(w["end"] * 1000),
                    ))

        text = result.get("text", "").strip()
        language = result.get("language", self.language or "en")

        # Estimate confidence from avg log prob
        avg_logprob = 0.0
        n_segs = 0
        for seg in result.get("segments", []):
            if "avg_logprob" in seg:
                avg_logprob += seg["avg_logprob"]
                n_segs += 1
        confidence = 0.0
        if n_segs > 0:
            avg_logprob /= n_segs
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

        return TranscriptResult(
            text=text,
            language=language,
            confidence=round(confidence, 3),
            words=words,
        )


class FasterWhisperTranscriber(BaseTranscriber):
    """Faster Whisper transcriber using CTranslate2.

    Install: pip install sonarwise[faster-whisper]
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "float16",
        word_timestamps: bool = True,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.word_timestamps = word_timestamps
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                f"Faster Whisper {self.model_size} loaded "
                f"(compute_type={self.compute_type})"
            )
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install sonarwise[faster-whisper]"
            )

    def transcribe(self, audio: AudioSegment) -> TranscriptResult:
        self._load_model()

        samples = audio.audio.astype(np.float32)

        segments_gen, info = self._model.transcribe(
            samples,
            language=self.language,
            word_timestamps=self.word_timestamps,
        )

        text_parts = []
        words = []
        total_logprob = 0.0
        n_segs = 0

        for seg in segments_gen:
            text_parts.append(seg.text.strip())
            total_logprob += seg.avg_logprob
            n_segs += 1
            if self.word_timestamps and seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word.strip(),
                        start_ms=int(w.start * 1000),
                        end_ms=int(w.end * 1000),
                    ))

        confidence = 0.0
        if n_segs > 0:
            confidence = min(1.0, max(0.0, 1.0 + total_logprob / n_segs))

        return TranscriptResult(
            text=" ".join(text_parts),
            language=info.language or self.language or "en",
            confidence=round(confidence, 3),
            words=words,
        )
