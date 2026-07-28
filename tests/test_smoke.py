"""Smoke tests — end-to-end pipeline verification.

These tests verify that the full SonarWise pipeline works end-to-end
using dummy/mock components. No GPU or heavy model downloads required.
"""

import os
import tempfile
import struct

import numpy as np
import pytest

from sonarwise.core.pipeline import SonarWise
from sonarwise.core.transcriber import BaseTranscriber
from sonarwise.core.embedder import BaseAudioEmbedder
from sonarwise.core.chunker import BaseChunker, FixedChunker
from sonarwise.core.diarizer import BaseDiarizer
from sonarwise.core.speaker_embedder import BaseSpeakerEmbedder
from sonarwise.core.event_classifier import BaseEventClassifier
from sonarwise.core.store import SQLiteStore
from sonarwise.core.models import (
    AudioData, AudioSegment, TranscriptResult, Word,
    EventTag, SpeakerSegment, Segment,
)


# ─── Dummy Components ──────────────────────────────────────────────

class DummyTranscriber(BaseTranscriber):
    """Returns predictable transcriptions for testing."""

    def __init__(self):
        self._call_count = 0
        self._transcripts = [
            "we need to cut the budget by fifteen percent",
            "the auth service migration is on track",
            "any blockers for this sprint",
            "database team hasnt given us access yet",
            "ill follow up with them today",
        ]

    def transcribe(self, audio: AudioSegment) -> TranscriptResult:
        text = self._transcripts[self._call_count % len(self._transcripts)]
        self._call_count += 1
        return TranscriptResult(
            text=text,
            language="en",
            confidence=0.92,
            words=[
                Word(text=w, start_ms=i * 200, end_ms=(i + 1) * 200)
                for i, w in enumerate(text.split())
            ],
        )


class DummyEmbedder(BaseAudioEmbedder):
    """Returns deterministic embeddings based on content hash."""

    def embed_audio(self, audio: AudioSegment) -> np.ndarray:
        np.random.seed(int(audio.start_ms) % 10000)
        emb = np.random.randn(512).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def embed_text(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 10000)
        emb = np.random.randn(512).astype(np.float32)
        return emb / np.linalg.norm(emb)


class DummyDiarizer(BaseDiarizer):
    """Assigns alternating speakers."""

    def diarize(self, audio: AudioData) -> list:
        duration = audio.duration_ms
        segments = []
        chunk_ms = 5000
        speaker_idx = 0
        pos = 0
        while pos < duration:
            end = min(pos + chunk_ms, duration)
            segments.append(SpeakerSegment(
                start_ms=pos,
                end_ms=end,
                speaker_id=f"speaker_{speaker_idx % 2}",
            ))
            speaker_idx += 1
            pos = end
        return segments


class DummySpeakerEmbedder(BaseSpeakerEmbedder):
    """Returns fixed embeddings per speaker."""

    def embed(self, audio: AudioSegment) -> np.ndarray:
        np.random.seed(42)
        emb = np.random.randn(192).astype(np.float32)
        return emb / np.linalg.norm(emb)


class DummyEventClassifier(BaseEventClassifier):
    """Classifies based on audio energy."""

    def classify(self, audio: AudioSegment) -> list:
        rms = float(np.sqrt(np.mean(audio.audio ** 2)))
        if rms < 0.01:
            return [EventTag(label="silence", confidence=0.95)]
        return [EventTag(label="speech", confidence=0.85)]


# ─── Helpers ────────────────────────────────────────────────────────

def _create_test_wav(filepath: str, duration_s: float = 5.0, sr: int = 16000):
    """Create a valid WAV file with random audio data."""
    n_samples = int(duration_s * sr)
    # Mix of silence and speech-like noise
    samples = np.zeros(n_samples, dtype=np.float32)
    # Add speech-like segments
    chunk = n_samples // 5
    for i in range(0, n_samples, chunk * 2):
        end = min(i + chunk, n_samples)
        samples[i:end] = np.random.randn(end - i) * 0.3

    samples_int16 = (samples * 32767).astype(np.int16)
    data = samples_int16.tobytes()

    with open(filepath, "wb") as f:
        n_channels = 1
        sample_width = 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * n_channels * sample_width))
        f.write(struct.pack("<H", n_channels * sample_width))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def test_wav(tmp_dir):
    """Create a test WAV file."""
    path = os.path.join(tmp_dir, "test.wav")
    _create_test_wav(path, duration_s=10.0)
    return path


@pytest.fixture
def test_wav_short(tmp_dir):
    """Create a short test WAV file (for audio query)."""
    path = os.path.join(tmp_dir, "clip.wav")
    _create_test_wav(path, duration_s=2.0)
    return path


@pytest.fixture
def test_folder(tmp_dir):
    """Create a folder with multiple test WAV files."""
    for name in ["meeting_1.wav", "meeting_2.wav", "call.wav"]:
        _create_test_wav(os.path.join(tmp_dir, name), duration_s=5.0)
    return tmp_dir


@pytest.fixture
def db_path(tmp_dir):
    """Temporary database path."""
    return os.path.join(tmp_dir, "test.db")


@pytest.fixture
def sw(db_path):
    """SonarWise instance with all dummy components."""
    instance = SonarWise(
        transcriber=DummyTranscriber(),
        embedder=DummyEmbedder(),
        chunker=FixedChunker(window_ms=2000),
        diarizer=DummyDiarizer(),
        speaker_embedder=DummySpeakerEmbedder(),
        event_classifier=DummyEventClassifier(),
        diarization=True,
        events=True,
        db_path=db_path,
    )
    yield instance
    instance.close()


# ─── Smoke Tests ────────────────────────────────────────────────────

class TestSmokeIndex:
    """Test indexing pipeline end-to-end."""

    def test_index_single_file(self, sw, test_wav):
        count = sw.index(test_wav)
        assert count > 0

    def test_index_returns_segment_count(self, sw, test_wav):
        count = sw.index(test_wav)
        assert count == sw._store.count()

    def test_index_folder(self, sw, test_folder):
        count = sw.index_folder(test_folder)
        assert count > 0
        files = sw._store.list_files()
        assert len(files) == 3

    def test_index_folder_recursive(self, sw, tmp_dir):
        sub = os.path.join(tmp_dir, "sub")
        os.makedirs(sub)
        _create_test_wav(os.path.join(tmp_dir, "root.wav"))
        _create_test_wav(os.path.join(sub, "nested.wav"))
        count = sw.index_folder(tmp_dir, recursive=True)
        assert count > 0
        files = sw._store.list_files()
        assert len(files) == 2

    def test_index_skip_existing(self, sw, test_wav):
        sw.index(test_wav)
        count1 = sw._store.count()
        sw.index(test_wav, skip_existing=True)
        count2 = sw._store.count()
        assert count1 == count2

    def test_reindex_replaces(self, sw, test_wav):
        sw.index(test_wav)
        count1 = sw._store.count()
        sw.index(test_wav)  # re-index (no skip)
        count2 = sw._store.count()
        assert count1 == count2  # should replace, not duplicate


class TestSmokeQuery:
    """Test query pipeline end-to-end."""

    def test_text_query(self, sw, test_wav):
        sw.index(test_wav)
        results = sw.query("budget", top_k=3)
        assert len(results) > 0
        assert results[0].transcript != ""
        assert results[0].score > 0

    def test_text_query_returns_metadata(self, sw, test_wav):
        sw.index(test_wav)
        results = sw.query("budget", top_k=1)
        r = results[0]
        assert r.filepath == os.path.abspath(test_wav)
        assert r.start_ms >= 0
        assert r.end_ms > r.start_ms
        assert r.duration_ms > 0
        assert isinstance(r.score, float)

    def test_audio_query(self, sw, test_wav, test_wav_short):
        sw.index(test_wav)
        results = sw.query_audio(test_wav_short, top_k=3)
        assert len(results) > 0

    def test_speaker_filtered_query(self, sw, test_wav):
        sw.index(test_wav)
        results = sw.query("budget", speaker="speaker_0", top_k=5)
        for r in results:
            assert r.speaker_id == "speaker_0"

    def test_event_query(self, sw, test_wav):
        sw.index(test_wav)
        results = sw.query_events(event="speech", top_k=5)
        assert len(results) > 0
        for r in results:
            assert "speech" in r.event_tags

    def test_empty_query(self, sw):
        results = sw.query("anything", top_k=5)
        assert results == []

    def test_top_k_limit(self, sw, test_wav):
        sw.index(test_wav)
        results = sw.query("budget", top_k=2)
        assert len(results) <= 2


class TestSmokeSpeaker:
    """Test speaker management end-to-end."""

    def test_get_speakers(self, sw, test_wav):
        sw.index(test_wav)
        speakers = sw.get_speakers(test_wav)
        assert len(speakers) > 0
        for s in speakers:
            assert s.duration_ms > 0
            assert s.segment_count > 0

    def test_speaker_timeline(self, sw, test_wav):
        sw.index(test_wav)
        timeline = sw.speaker_timeline(test_wav)
        assert len(timeline) > 0
        for speaker, ranges in timeline.items():
            assert len(ranges) > 0
            for start, end in ranges:
                assert end > start

    def test_speaker_stats(self, sw, test_wav):
        sw.index(test_wav)
        stats = sw.speaker_stats(test_wav)
        assert len(stats) > 0
        total_pct = sum(s.percentage for s in stats.values())
        assert 99.0 <= total_pct <= 101.0  # should add up to ~100%

    def test_interaction_matrix(self, sw, test_wav):
        sw.index(test_wav)
        matrix = sw.interaction_matrix(test_wav)
        # With alternating speakers, there should be transitions
        assert isinstance(matrix, dict)

    def test_register_speaker(self, sw, test_wav):
        sw.register_speaker("TestSpeaker", reference_audio=test_wav)
        speakers = sw.list_registered_speakers()
        assert len(speakers) == 1
        assert speakers[0].name == "TestSpeaker"

    def test_remove_speaker(self, sw, test_wav):
        sw.register_speaker("TestSpeaker", reference_audio=test_wav)
        sw.remove_speaker("TestSpeaker")
        speakers = sw.list_registered_speakers()
        assert len(speakers) == 0

    def test_find_speaker_across(self, sw, test_folder):
        sw.index_folder(test_folder)
        result = sw.find_speaker_across(speaker="speaker_0")
        assert result["files_found_in"] > 0
        assert result["total_segments"] > 0


class TestSmokeManagement:
    """Test management operations."""

    def test_list_sources(self, sw, test_wav):
        sw.index(test_wav)
        sources = sw.list_sources()
        assert len(sources) == 1
        assert sources[0].segment_count > 0
        assert sources[0].duration_ms > 0

    def test_remove_file(self, sw, test_wav):
        sw.index(test_wav)
        assert sw._store.count() > 0
        n = sw.remove(test_wav)
        assert n > 0
        assert sw._store.count() == 0

    def test_remove_all(self, sw, test_folder):
        sw.index_folder(test_folder)
        assert sw._store.count() > 0
        n = sw.remove_all()
        assert n > 0
        assert sw._store.count() == 0

    def test_stats(self, sw, test_wav):
        sw.index(test_wav)
        stats = sw.stats()
        assert stats["total_files"] == 1
        assert stats["total_segments"] > 0
        assert stats["total_duration_ms"] > 0
        assert "en" in stats["unique_languages"]
        assert isinstance(stats["event_distribution"], dict)

    def test_stats_empty(self, sw):
        stats = sw.stats()
        assert stats["total_files"] == 0
        assert stats["total_segments"] == 0

    def test_context_manager(self, db_path, test_wav):
        with SonarWise(
            transcriber=DummyTranscriber(),
            embedder=DummyEmbedder(),
            chunker=FixedChunker(window_ms=2000),
            event_classifier=DummyEventClassifier(),
            db_path=db_path,
        ) as sw:
            sw.index(test_wav)
            assert sw._store.count() > 0

    def test_repr(self, sw):
        r = repr(sw)
        assert "SonarWise" in r
        assert "diarization=True" in r


class TestSmokeExport:
    """Test export functionality."""

    def test_export_json(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="json", output=os.path.join(tmp_dir, "out.json"))
        assert os.path.exists(out)
        import json
        with open(out) as f:
            data = json.load(f)
        assert len(data) > 0
        assert "transcript" in data[0]
        assert "start_ms" in data[0]

    def test_export_csv(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="csv", output=os.path.join(tmp_dir, "out.csv"))
        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert len(lines) > 1  # header + data

    def test_export_srt(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="srt", output=os.path.join(tmp_dir, "out.srt"))
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "-->" in content  # SRT timestamp format

    def test_export_vtt(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="vtt", output=os.path.join(tmp_dir, "out.vtt"))
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "WEBVTT" in content

    def test_export_txt(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="txt", output=os.path.join(tmp_dir, "out.txt"))
        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert len(lines) > 0

    def test_export_notes(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        out = sw.export(test_wav, format="notes", output=os.path.join(tmp_dir, "notes.md"))
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "## Meeting" in content
        assert "Speaker Summary" in content

    def test_export_invalid_format(self, sw, test_wav, tmp_dir):
        sw.index(test_wav)
        with pytest.raises(ValueError, match="Unknown format"):
            sw.export(test_wav, format="xml")

    def test_export_no_segments(self, sw, tmp_dir):
        with pytest.raises(ValueError, match="No segments found"):
            sw.export("/nonexistent/file.wav", format="json")


class TestSmokeEdgeCases:
    """Test edge cases and error handling."""

    def test_index_nonexistent_file(self, sw):
        with pytest.raises(FileNotFoundError):
            sw.index("/nonexistent/audio.wav")

    def test_index_empty_folder(self, sw, tmp_dir):
        count = sw.index_folder(tmp_dir)
        assert count == 0

    def test_query_empty_index(self, sw):
        results = sw.query("anything")
        assert results == []

    def test_query_events_empty_index(self, sw):
        results = sw.query_events("alarm")
        assert results == []

    def test_multiple_indexes_same_file(self, sw, test_wav):
        sw.index(test_wav)
        count1 = sw._store.count()
        sw.index(test_wav)
        count2 = sw._store.count()
        # Re-indexing should replace, not append
        assert count1 == count2

    def test_progress_callback(self, sw, test_folder):
        progress_calls = []
        sw.index_folder(
            test_folder,
            on_progress=lambda p: progress_calls.append(p),
        )
        assert len(progress_calls) == 3  # 3 files
        assert progress_calls[0]["index"] == 1
        assert progress_calls[-1]["index"] == 3

    def test_file_specific_query(self, sw, test_folder):
        sw.index_folder(test_folder)
        files = sw._store.list_files()
        results = sw.query("budget", filepath=files[0], top_k=10)
        for r in results:
            assert r.filepath == files[0]


class TestSmokeLiveMode:
    """Test live mode with file stream listener (simulated)."""

    def test_file_stream_listener(self, sw, test_wav):
        from sonarwise.core.stream_listener import FileStreamListener

        listener = FileStreamListener(filepath=test_wav, chunk_duration_ms=2000)
        listener.start()

        chunks = []
        for chunk in listener.stream():
            chunks.append(chunk)
            if len(chunks) >= 3:
                break

        listener.stop()
        assert len(chunks) >= 3
        assert chunks[0].sample_rate == 16000
        assert chunks[0].timestamp_ms == 0
        assert chunks[1].timestamp_ms == 2000

    def test_callback_registration(self, sw):
        called = []

        @sw.on("transcript")
        def on_transcript(seg):
            called.append(seg)

        assert "transcript" in sw._callbacks
        assert len(sw._callbacks["transcript"]) == 1

    def test_keyword_callback_registration(self, sw):
        @sw.on("keyword", words=["budget", "risk"])
        def on_kw(seg):
            pass

        cb = sw._callbacks["keyword"][0]
        assert cb["kwargs"]["words"] == ["budget", "risk"]

    def test_multiple_callbacks(self, sw):
        @sw.on("transcript")
        def cb1(seg):
            pass

        @sw.on("transcript")
        def cb2(seg):
            pass

        @sw.on("keyword", words=["test"])
        def cb3(seg):
            pass

        assert len(sw._callbacks["transcript"]) == 2
        assert len(sw._callbacks["keyword"]) == 1
