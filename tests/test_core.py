"""Tests for sonarwise core components."""

import os
import tempfile

import numpy as np
import pytest

from sonarwise.core.models import (
    AudioData, AudioSegment, EventTag, Segment, SearchResult, Word,
    TranscriptResult, SpeakerSegment,
)
from sonarwise.core.chunker import FixedChunker, EnergyChunker
from sonarwise.core.store import SQLiteStore
from sonarwise.core.event_classifier import SimpleEventClassifier
from sonarwise.core.transcriber import BaseTranscriber
from sonarwise.core.embedder import BaseAudioEmbedder
from sonarwise.utils.similarity import cosine_similarity, cosine_similarity_batch
from sonarwise.utils.time_utils import ms_to_timestamp, ms_to_srt_timestamp, ms_to_short
from sonarwise.config import SonarWiseConfig


# ─── Models ─────────────────────────────────────────────────────────

class TestModels:
    def test_audio_data(self):
        samples = np.zeros(16000, dtype=np.float32)
        ad = AudioData(samples=samples, sample_rate=16000)
        assert ad.duration_ms == 1000

    def test_audio_segment(self):
        seg = AudioSegment(
            audio=np.zeros(8000, dtype=np.float32),
            sample_rate=16000,
            start_ms=0,
            end_ms=500,
        )
        assert seg.duration_ms == 500

    def test_segment_defaults(self):
        seg = Segment(filepath="test.wav", start_ms=0, end_ms=5000)
        assert seg.duration_ms == 5000
        assert seg.source_type == "batch"
        assert len(seg.segment_id) == 12


# ─── Chunker ────────────────────────────────────────────────────────

class TestChunker:
    def test_fixed_chunker(self):
        samples = np.random.randn(48000).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000)
        chunker = FixedChunker(window_ms=1000)
        segments = chunker.chunk(audio)
        assert len(segments) == 3
        assert segments[0].duration_ms == 1000

    def test_fixed_chunker_with_overlap(self):
        samples = np.random.randn(32000).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000)
        chunker = FixedChunker(window_ms=1000, overlap_ms=500)
        segments = chunker.chunk(audio)
        assert len(segments) >= 3

    def test_energy_chunker_silence(self):
        # All silence
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000)
        chunker = EnergyChunker(min_segment_ms=100)
        segments = chunker.chunk(audio)
        assert len(segments) == 0

    def test_energy_chunker_with_speech(self):
        # Silence + speech + silence
        samples = np.zeros(48000, dtype=np.float32)
        samples[16000:32000] = np.random.randn(16000) * 0.5
        audio = AudioData(samples=samples, sample_rate=16000)
        chunker = EnergyChunker(min_segment_ms=100)
        segments = chunker.chunk(audio)
        assert len(segments) >= 1


# ─── SQLiteStore ────────────────────────────────────────────────────

class TestSQLiteStore:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = SQLiteStore(db_path=self.tmp.name)

    def teardown_method(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_insert_and_count(self):
        seg = Segment(
            filepath="test.wav",
            start_ms=0,
            end_ms=5000,
            transcript="hello world",
            embedding=np.random.randn(512).tolist(),
        )
        self.store.insert(seg)
        assert self.store.count() == 1

    def test_search(self):
        emb = np.random.randn(512).astype(np.float32)
        emb_normalized = emb / np.linalg.norm(emb)

        seg = Segment(
            filepath="test.wav",
            start_ms=0,
            end_ms=5000,
            transcript="budget discussion",
            embedding=emb_normalized.tolist(),
        )
        self.store.insert(seg)

        results = self.store.search(emb_normalized, top_k=5)
        assert len(results) == 1
        assert results[0].score > 0.99

    def test_search_with_filters(self):
        base_emb = np.random.randn(512).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)
        for i in range(3):
            # All embeddings point in similar direction so scores > 0
            noise = np.random.randn(512).astype(np.float32) * 0.1
            emb = base_emb + noise
            emb = emb / np.linalg.norm(emb)
            seg = Segment(
                filepath="test.wav",
                start_ms=i * 5000,
                end_ms=(i + 1) * 5000,
                transcript=f"segment {i}",
                embedding=emb.tolist(),
                speaker_name="Ant" if i < 2 else "Other",
            )
            self.store.insert(seg)

        results = self.store.search(
            base_emb, top_k=10, filters={"speaker_name": "Ant"}
        )
        assert len(results) == 2

    def test_delete(self):
        seg = Segment(filepath="test.wav", start_ms=0, end_ms=5000,
                      embedding=np.random.randn(512).tolist())
        sid = self.store.insert(seg)
        assert self.store.count() == 1
        self.store.delete(sid)
        assert self.store.count() == 0

    def test_delete_by_file(self):
        for i in range(5):
            seg = Segment(filepath="test.wav", start_ms=i * 1000, end_ms=(i + 1) * 1000,
                          embedding=np.random.randn(512).tolist())
            self.store.insert(seg)
        assert self.store.count() == 5
        n = self.store.delete_by_file("test.wav")
        assert n == 5
        assert self.store.count() == 0

    def test_list_files(self):
        for fp in ["a.wav", "b.wav", "a.wav"]:
            seg = Segment(filepath=fp, start_ms=0, end_ms=1000,
                          embedding=np.random.randn(512).tolist())
            self.store.insert(seg)
        files = self.store.list_files()
        assert set(files) == {"a.wav", "b.wav"}

    def test_speaker_registry(self):
        emb = np.random.randn(192).tolist()
        self.store.register_speaker("Ant", emb, "2026-07-27")
        speakers = self.store.get_registered_speakers()
        assert len(speakers) == 1
        assert speakers[0]["name"] == "Ant"

        self.store.remove_speaker("Ant")
        speakers = self.store.get_registered_speakers()
        assert len(speakers) == 0


# ─── Similarity ─────────────────────────────────────────────────────

class TestSimilarity:
    def test_cosine_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_batch_similarity(self):
        query = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        scores = cosine_similarity_batch(query, embeddings)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(-1.0)


# ─── Time Utils ─────────────────────────────────────────────────────

class TestTimeUtils:
    def test_ms_to_timestamp(self):
        assert ms_to_timestamp(0) == "0:00.000"
        assert ms_to_timestamp(61000) == "1:01.000"
        assert ms_to_timestamp(3661500) == "1:01:01.500"

    def test_ms_to_srt(self):
        assert ms_to_srt_timestamp(61500) == "00:01:01,500"

    def test_ms_to_short(self):
        assert ms_to_short(0) == "0:00"
        assert ms_to_short(61000) == "1:01"
        assert ms_to_short(3661000) == "1:01:01"


# ─── Event Classifier ──────────────────────────────────────────────

class TestEventClassifier:
    def test_simple_silence(self):
        clf = SimpleEventClassifier()
        seg = AudioSegment(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000, start_ms=0, end_ms=1000,
        )
        events = clf.classify(seg)
        assert any(e.label == "silence" for e in events)

    def test_simple_speech(self):
        clf = SimpleEventClassifier()
        seg = AudioSegment(
            audio=np.random.randn(16000).astype(np.float32) * 0.3,
            sample_rate=16000, start_ms=0, end_ms=1000,
        )
        events = clf.classify(seg)
        assert len(events) >= 1


# ─── Config ─────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = SonarWiseConfig()
        assert cfg.transcriber_model == "base"
        assert cfg.db_path == "sonarwise.db"
        assert cfg.diarization is False

    def test_yaml_roundtrip(self):
        cfg = SonarWiseConfig(transcriber_model="large-v3", diarization=True)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        cfg.to_yaml(path)
        loaded = SonarWiseConfig.from_yaml(path)
        os.unlink(path)
        assert loaded.diarization is True


# ─── Custom Pluggable Components ────────────────────────────────────

class TestPluggable:
    def test_custom_transcriber(self):
        class DummyTranscriber(BaseTranscriber):
            def transcribe(self, audio):
                return TranscriptResult(text="dummy text", language="en", confidence=1.0)

        t = DummyTranscriber()
        seg = AudioSegment(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000, start_ms=0, end_ms=1000,
        )
        result = t.transcribe(seg)
        assert result.text == "dummy text"

    def test_custom_embedder(self):
        class DummyEmbedder(BaseAudioEmbedder):
            def embed_audio(self, audio):
                return np.random.randn(512).astype(np.float32)

            def embed_text(self, text):
                return np.random.randn(512).astype(np.float32)

        e = DummyEmbedder()
        seg = AudioSegment(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000, start_ms=0, end_ms=1000,
        )
        emb = e.embed_audio(seg)
        assert emb.shape == (512,)
