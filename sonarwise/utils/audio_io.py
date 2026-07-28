"""Audio I/O utilities — decode, resample, normalize any audio format."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

from sonarwise.core.models import AudioData, AudioInfo

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "opus", "webm", "amr"}


class AudioIO:
    """Load, save, and inspect audio files. Uses ffmpeg for format conversion."""

    @staticmethod
    def load(
        filepath: str,
        target_sr: int = 16000,
        mono: bool = True,
    ) -> AudioData:
        """Load audio file, resample to target_sr, convert to mono float32.

        Args:
            filepath: Path to audio file.
            target_sr: Target sample rate (default 16kHz).
            mono: Convert to mono (default True).

        Returns:
            AudioData with normalized float32 samples.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: .{ext}. Supported: {SUPPORTED_FORMATS}")

        samples = AudioIO._load_with_ffmpeg(filepath, target_sr, mono)

        return AudioData(
            samples=samples,
            sample_rate=target_sr,
            channels=1 if mono else 0,
            source_path=os.path.abspath(filepath),
        )

    @staticmethod
    def _load_with_ffmpeg(
        filepath: str,
        target_sr: int,
        mono: bool,
    ) -> np.ndarray:
        """Decode audio using ffmpeg subprocess."""
        channels = "1" if mono else "2"
        cmd = [
            "ffmpeg",
            "-i", filepath,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(target_sr),
            "-ac", channels,
            "-loglevel", "error",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, check=True, timeout=300
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it: sudo apt install ffmpeg (Linux) "
                "or brew install ffmpeg (macOS)"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error decoding {filepath}: {e.stderr.decode()}")

        samples = np.frombuffer(result.stdout, dtype=np.float32)

        if len(samples) == 0:
            raise ValueError(f"No audio data decoded from {filepath}")

        # Normalize to [-1.0, 1.0]
        max_val = np.abs(samples).max()
        if max_val > 0:
            samples = samples / max_val

        return samples

    @staticmethod
    def load_from_bytes(
        data: bytes,
        format: str = "wav",
        target_sr: int = 16000,
        mono: bool = True,
    ) -> AudioData:
        """Load audio from raw bytes."""
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            return AudioIO.load(tmp_path, target_sr=target_sr, mono=mono)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def save(audio: AudioData, filepath: str, bitrate: str = "192k") -> None:
        """Save audio data to file."""
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            AudioIO._write_wav(f, audio)

        if ext == "wav":
            os.rename(tmp_path, filepath)
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-b:a", bitrate,
                "-loglevel", "error",
                filepath,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        logger.info(f"Saved audio to {filepath}")

    @staticmethod
    def _write_wav(f, audio: AudioData) -> None:
        """Write raw WAV file."""
        import struct

        samples = (audio.samples * 32767).astype(np.int16)
        data = samples.tobytes()
        n_channels = audio.channels or 1
        sample_width = 2  # 16-bit
        frame_rate = audio.sample_rate

        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", frame_rate))
        f.write(struct.pack("<I", frame_rate * n_channels * sample_width))
        f.write(struct.pack("<H", n_channels * sample_width))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)

    @staticmethod
    def info(filepath: str) -> AudioInfo:
        """Get audio file metadata without loading full audio."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            filepath,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except FileNotFoundError:
            raise RuntimeError("ffprobe not found. Install ffmpeg.")

        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        audio_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "audio":
                audio_stream = s
                break

        duration_s = float(fmt.get("duration", 0))
        sr = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
        channels = int(audio_stream.get("channels", 0)) if audio_stream else 0
        size_bytes = int(fmt.get("size", 0))

        ext = os.path.splitext(filepath)[1].lower().lstrip(".")

        return AudioInfo(
            duration_ms=int(duration_s * 1000),
            sample_rate=sr,
            channels=channels,
            format=ext,
            size_mb=round(size_bytes / (1024 * 1024), 2),
        )
