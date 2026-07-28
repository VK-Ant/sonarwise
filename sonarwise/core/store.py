"""Vector store — storage and similarity search for audio segments."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from sonarwise.core.models import SearchResult, Segment
from sonarwise.utils.similarity import cosine_similarity_batch

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Base class for vector stores. Implement to bring your own backend."""

    @abstractmethod
    def insert(self, segment: Segment) -> str:
        """Insert a segment. Returns segment_id."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search by embedding similarity.

        Args:
            embedding: Query vector.
            top_k: Max results to return.
            threshold: Minimum similarity score.
            filters: Optional filters (speaker_name, filepath, event_tags, etc.)

        Returns:
            List of SearchResult sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, segment_id: str) -> bool:
        """Delete a segment by ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_file(self, filepath: str) -> int:
        """Delete all segments from a file. Returns count deleted."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total number of stored segments."""
        raise NotImplementedError

    @abstractmethod
    def list_files(self) -> List[str]:
        """List all indexed filepaths."""
        raise NotImplementedError

    @abstractmethod
    def get_segments_by_file(self, filepath: str) -> List[Segment]:
        """Get all segments for a file."""
        raise NotImplementedError

    def close(self):
        """Clean up resources."""
        pass


class SQLiteStore(BaseVectorStore):
    """SQLite-based vector store. Zero setup, single file database.

    Embeddings stored as binary blobs. Similarity computed in Python.
    Good for small-to-medium datasets (< 100k segments).
    """

    def __init__(self, db_path: str = "sonarwise.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                transcript TEXT DEFAULT '',
                embedding BLOB,
                speaker_id TEXT,
                speaker_name TEXT,
                speaker_embedding BLOB,
                event_tags TEXT DEFAULT '[]',
                event_confidence TEXT DEFAULT '[]',
                language TEXT DEFAULT 'en',
                word_timestamps TEXT DEFAULT '[]',
                source_type TEXT DEFAULT 'batch',
                indexed_at TEXT NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                name TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                registered_at TEXT NOT NULL
            )
        """)

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_filepath ON segments(filepath)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_speaker_name ON segments(speaker_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_type ON segments(source_type)"
        )
        self._conn.commit()
        logger.info(f"SQLiteStore initialized: {self.db_path}")

    def insert(self, segment: Segment) -> str:
        embedding_blob = None
        if segment.embedding is not None:
            embedding_blob = np.array(segment.embedding, dtype=np.float32).tobytes()

        speaker_emb_blob = None
        if segment.speaker_embedding is not None:
            speaker_emb_blob = np.array(
                segment.speaker_embedding, dtype=np.float32
            ).tobytes()

        word_ts = json.dumps([
            {"text": w.text, "start_ms": w.start_ms, "end_ms": w.end_ms}
            for w in segment.word_timestamps
        ])

        self._conn.execute(
            """INSERT OR REPLACE INTO segments
            (segment_id, filepath, start_ms, end_ms, duration_ms, transcript,
             embedding, speaker_id, speaker_name, speaker_embedding,
             event_tags, event_confidence, language, word_timestamps,
             source_type, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                segment.segment_id,
                segment.filepath,
                segment.start_ms,
                segment.end_ms,
                segment.duration_ms,
                segment.transcript,
                embedding_blob,
                segment.speaker_id,
                segment.speaker_name,
                speaker_emb_blob,
                json.dumps(segment.event_tags),
                json.dumps(segment.event_confidence),
                segment.language,
                word_ts,
                segment.source_type,
                segment.indexed_at,
            ),
        )
        self._conn.commit()
        return segment.segment_id

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query = "SELECT * FROM segments WHERE embedding IS NOT NULL"
        params: list = []

        if filters:
            if "filepath" in filters:
                query += " AND filepath = ?"
                params.append(filters["filepath"])
            if "speaker_name" in filters:
                query += " AND speaker_name = ?"
                params.append(filters["speaker_name"])
            if "speaker_id" in filters:
                query += " AND speaker_id = ?"
                params.append(filters["speaker_id"])
            if "language" in filters:
                query += " AND language = ?"
                params.append(filters["language"])
            if "source_type" in filters:
                query += " AND source_type = ?"
                params.append(filters["source_type"])

        rows = self._conn.execute(query, params).fetchall()

        if not rows:
            return []

        query_emb = np.asarray(embedding, dtype=np.float32)

        # Build embedding matrix
        embeddings = []
        valid_rows = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(emb)
            valid_rows.append(row)

        if not embeddings:
            return []

        emb_matrix = np.stack(embeddings)
        scores = cosine_similarity_batch(query_emb, emb_matrix)

        # Filter by event_tags if specified
        event_filter = filters.get("event_tags") if filters else None

        results = []
        for i, (row, score) in enumerate(zip(valid_rows, scores)):
            if score < threshold:
                continue

            event_tags = json.loads(row["event_tags"])
            if event_filter:
                if not any(e in event_tags for e in event_filter):
                    continue

            word_ts_raw = json.loads(row["word_timestamps"])
            from sonarwise.core.models import Word
            word_ts = [
                Word(text=w["text"], start_ms=w["start_ms"], end_ms=w["end_ms"])
                for w in word_ts_raw
            ]

            results.append(SearchResult(
                segment_id=row["segment_id"],
                filepath=row["filepath"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                duration_ms=row["duration_ms"],
                transcript=row["transcript"],
                speaker_id=row["speaker_id"],
                speaker_name=row["speaker_name"],
                event_tags=event_tags,
                event_confidence=json.loads(row["event_confidence"]),
                language=row["language"],
                word_timestamps=word_ts,
                score=round(float(score), 4),
                source_type=row["source_type"],
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, segment_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM segments WHERE segment_id = ?", (segment_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_by_file(self, filepath: str) -> int:
        cursor = self._conn.execute(
            "DELETE FROM segments WHERE filepath = ?", (filepath,)
        )
        self._conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM segments").fetchone()
        return row[0]

    def list_files(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT filepath FROM segments ORDER BY filepath"
        ).fetchall()
        return [r[0] for r in rows]

    def get_segments_by_file(self, filepath: str) -> List[Segment]:
        rows = self._conn.execute(
            "SELECT * FROM segments WHERE filepath = ? ORDER BY start_ms",
            (filepath,),
        ).fetchall()

        segments = []
        for row in rows:
            from sonarwise.core.models import Word
            word_ts_raw = json.loads(row["word_timestamps"])
            word_ts = [
                Word(text=w["text"], start_ms=w["start_ms"], end_ms=w["end_ms"])
                for w in word_ts_raw
            ]

            emb = None
            if row["embedding"]:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).tolist()

            spk_emb = None
            if row["speaker_embedding"]:
                spk_emb = np.frombuffer(
                    row["speaker_embedding"], dtype=np.float32
                ).tolist()

            segments.append(Segment(
                segment_id=row["segment_id"],
                filepath=row["filepath"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                duration_ms=row["duration_ms"],
                transcript=row["transcript"],
                embedding=emb,
                speaker_id=row["speaker_id"],
                speaker_name=row["speaker_name"],
                speaker_embedding=spk_emb,
                event_tags=json.loads(row["event_tags"]),
                event_confidence=json.loads(row["event_confidence"]),
                language=row["language"],
                word_timestamps=word_ts,
                source_type=row["source_type"],
                indexed_at=row["indexed_at"],
            ))
        return segments

    # --- Speaker registry ---

    def register_speaker(self, name: str, embedding: List[float], registered_at: str):
        blob = np.array(embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding, registered_at) VALUES (?, ?, ?)",
            (name, blob, registered_at),
        )
        self._conn.commit()

    def get_registered_speakers(self) -> List[dict]:
        rows = self._conn.execute("SELECT * FROM speakers").fetchall()
        result = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32).tolist()
            result.append({
                "name": r["name"],
                "embedding": emb,
                "registered_at": r["registered_at"],
            })
        return result

    def remove_speaker(self, name: str) -> bool:
        cursor = self._conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
