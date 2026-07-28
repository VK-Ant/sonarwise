#!/usr/bin/env python3
"""sonarwise v0.1.0 — Full project validation script.

Run: python validate.py

Tests everything without GPU or heavy model downloads.
Exit code 0 = all good, 1 = something failed.
"""

import os
import sys
import struct
import tempfile
import json

import numpy as np


def log(msg, status=""):
    icon = {"ok": "[PASS]", "fail": "[FAIL]", "info": "[INFO]", "": ""}
    print(f"  {icon.get(status, '')} {msg}")


def create_test_wav(filepath, duration_s=5.0, sr=16000):
    n_samples = int(duration_s * sr)
    samples = np.zeros(n_samples, dtype=np.float32)
    chunk = n_samples // 5
    for i in range(0, n_samples, chunk * 2):
        end = min(i + chunk, n_samples)
        samples[i:end] = np.random.randn(end - i) * 0.3

    samples_int16 = (samples * 32767).astype(np.int16)
    data = samples_int16.tobytes()

    with open(filepath, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def main():
    print("\n=== sonarwise v0.1.0 — Validation ===\n")
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "test.wav")
        wav_clip = os.path.join(tmp, "clip.wav")
        db_path = os.path.join(tmp, "test.db")

        # ── 1. Create test audio ─────────────────────────────────
        print("1. Audio I/O")
        try:
            create_test_wav(wav_path, duration_s=10.0)
            create_test_wav(wav_clip, duration_s=2.0)

            from sonarwise.utils.audio_io import AudioIO
            audio = AudioIO.load(wav_path)
            assert audio.sample_rate == 16000
            assert audio.duration_ms > 0
            assert len(audio.samples) > 0
            log(f"Load WAV: {audio.duration_ms}ms, {audio.sample_rate}Hz", "ok")
            passed += 1
        except Exception as e:
            log(f"Load WAV: {e}", "fail")
            failed += 1

        # ── 2. Chunking ──────────────────────────────────────────
        print("\n2. Chunking")
        try:
            from sonarwise.core.chunker import FixedChunker, EnergyChunker
            from sonarwise.core.models import AudioData

            chunker = FixedChunker(window_ms=2000)
            chunks = chunker.chunk(audio)
            assert len(chunks) > 0
            log(f"FixedChunker: {len(chunks)} segments", "ok")
            passed += 1

            energy_chunker = EnergyChunker(min_segment_ms=100)
            echunks = energy_chunker.chunk(audio)
            log(f"EnergyChunker: {len(echunks)} segments", "ok")
            passed += 1
        except Exception as e:
            log(f"Chunking: {e}", "fail")
            failed += 1

        # ── 3. SQLite Store ───────────────────────────────────────
        print("\n3. Vector Store")
        try:
            from sonarwise.core.store import SQLiteStore
            from sonarwise.core.models import Segment

            store = SQLiteStore(db_path=db_path)

            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            seg = Segment(
                filepath="test.wav", start_ms=0, end_ms=5000,
                transcript="hello world", embedding=emb.tolist(),
                speaker_id="speaker_0", event_tags=["speech"],
                event_confidence=[0.9],
            )
            store.insert(seg)
            assert store.count() == 1
            log("Insert segment", "ok")
            passed += 1

            results = store.search(emb, top_k=5)
            assert len(results) == 1
            assert results[0].score > 0.99
            log(f"Search: score={results[0].score}", "ok")
            passed += 1

            store.delete(seg.segment_id)
            assert store.count() == 0
            log("Delete segment", "ok")
            passed += 1

            store.close()
        except Exception as e:
            log(f"Store: {e}", "fail")
            failed += 1

        # ── 4. Similarity ────────────────────────────────────────
        print("\n4. Similarity")
        try:
            from sonarwise.utils.similarity import cosine_similarity, cosine_similarity_batch

            a = np.array([1.0, 0.0, 0.0])
            b = np.array([1.0, 0.0, 0.0])
            assert abs(cosine_similarity(a, b) - 1.0) < 0.001
            log("Cosine identical = 1.0", "ok")
            passed += 1

            a = np.array([1.0, 0.0])
            b = np.array([0.0, 1.0])
            assert abs(cosine_similarity(a, b)) < 0.001
            log("Cosine orthogonal = 0.0", "ok")
            passed += 1

            query = np.array([1.0, 0.0, 0.0])
            batch = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            scores = cosine_similarity_batch(query, batch)
            assert scores[0] > 0.99 and abs(scores[1]) < 0.01
            log("Batch similarity", "ok")
            passed += 1
        except Exception as e:
            log(f"Similarity: {e}", "fail")
            failed += 1

        # ── 5. Time Utils ────────────────────────────────────────
        print("\n5. Time Utils")
        try:
            from sonarwise.utils.time_utils import ms_to_short, ms_to_srt_timestamp

            assert ms_to_short(61000) == "1:01"
            assert ms_to_srt_timestamp(61500) == "00:01:01,500"
            log("Timestamp formatting", "ok")
            passed += 1
        except Exception as e:
            log(f"Time utils: {e}", "fail")
            failed += 1

        # ── 6. Config ────────────────────────────────────────────
        print("\n6. Configuration")
        try:
            from sonarwise.config import SonarWiseConfig

            cfg = SonarWiseConfig()
            assert cfg.transcriber_model == "base"
            assert cfg.db_path == "sonarwise.db"

            yaml_path = os.path.join(tmp, "test_config.yaml")
            cfg.to_yaml(yaml_path)
            loaded = SonarWiseConfig.from_yaml(yaml_path)
            assert loaded.transcriber_model == "base"
            log("Config create + YAML roundtrip", "ok")
            passed += 1
        except Exception as e:
            log(f"Config: {e}", "fail")
            failed += 1

        # ── 7. Full Pipeline (Dummy Components) ──────────────────
        print("\n7. Full Pipeline (dummy models)")
        try:
            from sonarwise.core.pipeline import SonarWise
            from sonarwise.core.transcriber import BaseTranscriber
            from sonarwise.core.embedder import BaseAudioEmbedder
            from sonarwise.core.diarizer import BaseDiarizer
            from sonarwise.core.event_classifier import BaseEventClassifier
            from sonarwise.core.models import (
                AudioSegment, TranscriptResult, EventTag, SpeakerSegment, AudioData,
            )

            class T(BaseTranscriber):
                def transcribe(self, a):
                    return TranscriptResult(text="test transcript", language="en", confidence=0.9)

            class E(BaseAudioEmbedder):
                def embed_audio(self, a):
                    e = np.random.randn(512).astype(np.float32)
                    return e / np.linalg.norm(e)
                def embed_text(self, t):
                    np.random.seed(hash(t) % 10000)
                    e = np.random.randn(512).astype(np.float32)
                    return e / np.linalg.norm(e)

            class D(BaseDiarizer):
                def diarize(self, a):
                    return [
                        SpeakerSegment(0, a.duration_ms // 2, "speaker_0"),
                        SpeakerSegment(a.duration_ms // 2, a.duration_ms, "speaker_1"),
                    ]

            class Ev(BaseEventClassifier):
                def classify(self, a):
                    return [EventTag(label="speech", confidence=0.9)]

            db2 = os.path.join(tmp, "pipeline.db")
            sw = SonarWise(
                transcriber=T(), embedder=E(),
                chunker=FixedChunker(window_ms=2000),
                diarizer=D(), event_classifier=Ev(),
                diarization=True, events=True, db_path=db2,
            )

            # Index
            count = sw.index(wav_path)
            assert count > 0
            log(f"Index: {count} segments", "ok")
            passed += 1

            # Text query
            results = sw.query("test", top_k=3)
            assert len(results) > 0
            log(f"Text query: {len(results)} results, top score={results[0].score}", "ok")
            passed += 1

            # Audio query
            results = sw.query_audio(wav_clip, top_k=3)
            assert len(results) > 0
            log(f"Audio query: {len(results)} results", "ok")
            passed += 1

            # Speaker query
            results = sw.query("test", speaker="speaker_0", top_k=5)
            for r in results:
                assert r.speaker_id == "speaker_0"
            log(f"Speaker query: {len(results)} results (filtered)", "ok")
            passed += 1

            # Event query
            events = sw.query_events("speech", top_k=5)
            assert len(events) > 0
            log(f"Event query: {len(events)} results", "ok")
            passed += 1

            # Speakers
            speakers = sw.get_speakers(wav_path)
            assert len(speakers) > 0
            log(f"Get speakers: {len(speakers)} speakers", "ok")
            passed += 1

            # Timeline
            timeline = sw.speaker_timeline(wav_path)
            assert len(timeline) > 0
            log(f"Speaker timeline: {len(timeline)} speakers", "ok")
            passed += 1

            # Stats
            stats = sw.stats()
            assert stats["total_files"] == 1
            assert stats["total_segments"] > 0
            log(f"Stats: {stats['total_segments']} segments, {stats['total_files']} files", "ok")
            passed += 1

            # Export JSON
            json_out = os.path.join(tmp, "export.json")
            sw.export(wav_path, format="json", output=json_out)
            with open(json_out) as f:
                data = json.load(f)
            assert len(data) > 0
            log(f"Export JSON: {len(data)} segments", "ok")
            passed += 1

            # Export SRT
            srt_out = os.path.join(tmp, "export.srt")
            sw.export(wav_path, format="srt", output=srt_out)
            with open(srt_out) as f:
                assert "-->" in f.read()
            log("Export SRT", "ok")
            passed += 1

            # Export notes
            notes_out = os.path.join(tmp, "notes.md")
            sw.export(wav_path, format="notes", output=notes_out)
            with open(notes_out) as f:
                assert "Speaker Summary" in f.read()
            log("Export notes", "ok")
            passed += 1

            # Remove
            n = sw.remove(wav_path)
            assert n > 0
            assert sw._store.count() == 0
            log(f"Remove: {n} segments deleted", "ok")
            passed += 1

            # List sources (should be empty now)
            sources = sw.list_sources()
            assert len(sources) == 0
            log("List sources: empty after remove", "ok")
            passed += 1

            sw.close()

        except Exception as e:
            log(f"Pipeline: {e}", "fail")
            failed += 1
            import traceback
            traceback.print_exc()

        # ── 8. Import Check ──────────────────────────────────────
        print("\n8. Import Check")
        try:
            from sonarwise import SonarWise, __version__, __tagline__
            assert __version__ == "0.1.0"
            assert __tagline__ == "Hear. Search. Retrieve."
            log(f"sonarwise v{__version__} — {__tagline__}", "ok")
            passed += 1
        except Exception as e:
            log(f"Import: {e}", "fail")
            failed += 1

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print(f"  TOTAL:  {passed + failed}")
    print(f"{'='*50}")

    if failed > 0:
        print("\n  Some tests failed. Check output above.\n")
        sys.exit(1)
    else:
        print("\n  All validations passed. Ready to ship.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
