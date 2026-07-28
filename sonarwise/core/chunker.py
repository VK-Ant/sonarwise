"""Audio chunking — split audio into segments for processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from sonarwise.core.models import AudioData, AudioSegment

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for audio chunkers. Implement chunk() to create a custom chunker."""

    @abstractmethod
    def chunk(self, audio: AudioData) -> List[AudioSegment]:
        """Split audio into segments.

        Args:
            audio: AudioData to split.

        Returns:
            List of AudioSegment with timing info.
        """
        raise NotImplementedError


class FixedChunker(BaseChunker):
    """Split audio into fixed-length windows with optional overlap."""

    def __init__(self, window_ms: int = 5000, overlap_ms: int = 0):
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms

    def chunk(self, audio: AudioData) -> List[AudioSegment]:
        samples = audio.samples
        sr = audio.sample_rate
        window_samples = int(self.window_ms * sr / 1000)
        step_samples = int((self.window_ms - self.overlap_ms) * sr / 1000)

        segments = []
        pos = 0
        while pos < len(samples):
            end = min(pos + window_samples, len(samples))
            chunk_audio = samples[pos:end]
            start_ms = int(pos / sr * 1000)
            end_ms = int(end / sr * 1000)

            segments.append(AudioSegment(
                audio=chunk_audio,
                sample_rate=sr,
                start_ms=start_ms,
                end_ms=end_ms,
            ))
            pos += step_samples

        logger.info(f"FixedChunker: {len(segments)} segments (window={self.window_ms}ms)")
        return segments


class EnergyChunker(BaseChunker):
    """Split audio by energy drops — silence detection via amplitude."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        min_segment_ms: int = 500,
        max_segment_ms: int = 30000,
        hop_ms: int = 20,
    ):
        self.energy_threshold = energy_threshold
        self.min_segment_ms = min_segment_ms
        self.max_segment_ms = max_segment_ms
        self.hop_ms = hop_ms

    def chunk(self, audio: AudioData) -> List[AudioSegment]:
        samples = audio.samples
        sr = audio.sample_rate
        hop_samples = int(self.hop_ms * sr / 1000)
        min_samples = int(self.min_segment_ms * sr / 1000)
        max_samples = int(self.max_segment_ms * sr / 1000)

        # Compute frame-level energy
        is_speech = []
        for i in range(0, len(samples), hop_samples):
            frame = samples[i : i + hop_samples]
            energy = np.sqrt(np.mean(frame**2))
            is_speech.append(energy > self.energy_threshold)

        segments = []
        in_segment = False
        seg_start = 0

        for i, active in enumerate(is_speech):
            sample_pos = i * hop_samples
            if active and not in_segment:
                seg_start = sample_pos
                in_segment = True
            elif not active and in_segment:
                seg_len = sample_pos - seg_start
                if seg_len >= min_samples:
                    segments.append(self._make_segment(samples, sr, seg_start, sample_pos))
                in_segment = False
            elif in_segment and (sample_pos - seg_start) >= max_samples:
                segments.append(self._make_segment(samples, sr, seg_start, sample_pos))
                seg_start = sample_pos

        # Handle last segment
        if in_segment:
            seg_len = len(samples) - seg_start
            if seg_len >= min_samples:
                segments.append(self._make_segment(samples, sr, seg_start, len(samples)))

        logger.info(f"EnergyChunker: {len(segments)} segments")
        return segments

    @staticmethod
    def _make_segment(
        samples: np.ndarray, sr: int, start: int, end: int
    ) -> AudioSegment:
        return AudioSegment(
            audio=samples[start:end],
            sample_rate=sr,
            start_ms=int(start / sr * 1000),
            end_ms=int(end / sr * 1000),
        )


class VADChunker(BaseChunker):
    """Voice Activity Detection chunker using Silero VAD.

    Falls back to EnergyChunker if Silero VAD is not installed.
    """

    def __init__(
        self,
        min_segment_ms: int = 500,
        max_segment_ms: int = 30000,
        silence_threshold_ms: int = 300,
        padding_ms: int = 200,
        threshold: float = 0.5,
    ):
        self.min_segment_ms = min_segment_ms
        self.max_segment_ms = max_segment_ms
        self.silence_threshold_ms = silence_threshold_ms
        self.padding_ms = padding_ms
        self.threshold = threshold
        self._model = None

    def _load_model(self):
        """Load Silero VAD model."""
        if self._model is not None:
            return
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            logger.info("Silero VAD model loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Silero VAD not available ({e}), falling back to EnergyChunker")
            self._model = None

    def chunk(self, audio: AudioData) -> List[AudioSegment]:
        self._load_model()

        if self._model is None:
            fallback = EnergyChunker(
                min_segment_ms=self.min_segment_ms,
                max_segment_ms=self.max_segment_ms,
            )
            return fallback.chunk(audio)

        return self._chunk_with_silero(audio)

    def _chunk_with_silero(self, audio: AudioData) -> List[AudioSegment]:
        import torch

        samples = audio.samples
        sr = audio.sample_rate

        # Silero expects 16kHz
        if sr != 16000:
            logger.warning(f"VAD expects 16kHz, got {sr}Hz. Results may vary.")

        tensor = torch.FloatTensor(samples)
        speech_timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_silence_duration_ms=self.silence_threshold_ms,
            min_speech_duration_ms=self.min_segment_ms,
        )

        segments = []
        pad_samples = int(self.padding_ms * sr / 1000)
        max_samples = int(self.max_segment_ms * sr / 1000)

        for ts in speech_timestamps:
            start = max(0, ts["start"] - pad_samples)
            end = min(len(samples), ts["end"] + pad_samples)

            # Force-split if too long
            while (end - start) > max_samples:
                split_end = start + max_samples
                segments.append(AudioSegment(
                    audio=samples[start:split_end],
                    sample_rate=sr,
                    start_ms=int(start / sr * 1000),
                    end_ms=int(split_end / sr * 1000),
                ))
                start = split_end

            if end > start:
                segments.append(AudioSegment(
                    audio=samples[start:end],
                    sample_rate=sr,
                    start_ms=int(start / sr * 1000),
                    end_ms=int(end / sr * 1000),
                ))

        logger.info(f"VADChunker: {len(segments)} segments from Silero VAD")
        return segments
