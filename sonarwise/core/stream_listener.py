"""Stream listener — capture live audio from microphone or remote sources."""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of live audio data."""

    data: np.ndarray
    sample_rate: int
    timestamp_ms: int = 0


class BaseStreamListener(ABC):
    """Base class for stream listeners. Implement to bring your own audio source."""

    @abstractmethod
    def start(self):
        """Start capturing audio."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stop capturing audio."""
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> Iterator[AudioChunk]:
        """Yield audio chunks as they arrive."""
        raise NotImplementedError


class MicListener(BaseStreamListener):
    """Capture audio from system microphone using sounddevice.

    Install: pip install sonarwise[live]
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 2000,
        channels: int = 1,
    ):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.channels = channels
        self._queue: queue.Queue = queue.Queue()
        self._stream = None
        self._running = False
        self._elapsed_ms = 0

    def start(self):
        """Start microphone capture."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed. Run: pip install sonarwise[live]"
            )

        chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic status: {status}")
            self._queue.put(indata.copy().flatten())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_samples,
            device=self.device_id,
            callback=callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            f"MicListener started (sr={self.sample_rate}, "
            f"chunk={self.chunk_duration_ms}ms)"
        )

    def stop(self):
        """Stop microphone capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("MicListener stopped")

    def stream(self) -> Iterator[AudioChunk]:
        """Yield audio chunks from microphone."""
        while self._running:
            try:
                data = self._queue.get(timeout=1.0)
                chunk = AudioChunk(
                    data=data.astype(np.float32),
                    sample_rate=self.sample_rate,
                    timestamp_ms=self._elapsed_ms,
                )
                self._elapsed_ms += self.chunk_duration_ms
                yield chunk
            except queue.Empty:
                continue


class FileStreamListener(BaseStreamListener):
    """Simulate live streaming from a file. Useful for testing."""

    def __init__(
        self,
        filepath: str,
        chunk_duration_ms: int = 2000,
        realtime: bool = False,
    ):
        self.filepath = filepath
        self.chunk_duration_ms = chunk_duration_ms
        self.realtime = realtime
        self._audio = None
        self._running = False

    def start(self):
        from sonarwise.utils.audio_io import AudioIO

        self._audio = AudioIO.load(self.filepath)
        self._running = True
        logger.info(f"FileStreamListener started: {self.filepath}")

    def stop(self):
        self._running = False
        logger.info("FileStreamListener stopped")

    def stream(self) -> Iterator[AudioChunk]:
        if self._audio is None:
            return

        sr = self._audio.sample_rate
        chunk_samples = int(self.chunk_duration_ms * sr / 1000)
        samples = self._audio.samples
        pos = 0
        elapsed_ms = 0

        while pos < len(samples) and self._running:
            end = min(pos + chunk_samples, len(samples))
            chunk_data = samples[pos:end]

            yield AudioChunk(
                data=chunk_data,
                sample_rate=sr,
                timestamp_ms=elapsed_ms,
            )

            elapsed_ms += self.chunk_duration_ms
            pos = end

            if self.realtime:
                import time
                time.sleep(self.chunk_duration_ms / 1000)
