"""SonarWise — main pipeline orchestrating all components.

Usage:
    from sonarwise import SonarWise

    sw = SonarWise(diarization=True, events=True)
    sw.index("meeting.wav")
    results = sw.query("budget discussion", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from sonarwise.config import SonarWiseConfig
from sonarwise.core.chunker import BaseChunker, VADChunker
from sonarwise.core.diarizer import BaseDiarizer, SimpleDiarizer
from sonarwise.core.embedder import BaseAudioEmbedder
from sonarwise.core.event_classifier import BaseEventClassifier, SimpleEventClassifier
from sonarwise.core.models import (
    AudioSegment as AudioSeg,
    EventResult,
    RegisteredSpeaker,
    SearchResult,
    Segment,
    Source,
    Speaker,
    SpeakerSegment,
    SpeakerStats,
)
from sonarwise.core.speaker_embedder import BaseSpeakerEmbedder
from sonarwise.core.store import BaseVectorStore, SQLiteStore
from sonarwise.core.stream_listener import AudioChunk, BaseStreamListener
from sonarwise.core.transcriber import BaseTranscriber
from sonarwise.utils.audio_io import AudioIO, SUPPORTED_FORMATS
from sonarwise.utils.similarity import cosine_similarity
from sonarwise.utils.time_utils import ms_to_short, ms_to_srt_timestamp, ms_to_vtt_timestamp

logger = logging.getLogger(__name__)


class SonarWise:
    """Pluggable audio perception engine. Hear. Search. Retrieve.

    Args:
        transcriber: Custom transcriber (default: WhisperTranscriber).
        embedder: Custom audio embedder (default: CLAPEmbedder).
        store: Custom vector store (default: SQLiteStore).
        chunker: Custom chunker (default: VADChunker).
        diarizer: Custom diarizer (default: SimpleDiarizer).
        speaker_embedder: Custom speaker embedder (default: None).
        event_classifier: Custom event classifier (default: SimpleEventClassifier).
        listener: Custom stream listener (default: None).
        diarization: Enable speaker diarization (default: False).
        events: Enable audio event detection (default: False).
        mode: "batch" or "live" (default: "batch").
        db_path: Path to SQLite database (default: "sonarwise.db").
        device: Compute device — "cuda", "cpu", "mps" (default: auto).
        verbose: Enable verbose logging (default: False).
        config: SonarWiseConfig object (overrides individual params).
    """

    def __init__(
        self,
        transcriber: Optional[BaseTranscriber] = None,
        embedder: Optional[BaseAudioEmbedder] = None,
        store: Optional[BaseVectorStore] = None,
        chunker: Optional[BaseChunker] = None,
        diarizer: Optional[BaseDiarizer] = None,
        speaker_embedder: Optional[BaseSpeakerEmbedder] = None,
        event_classifier: Optional[BaseEventClassifier] = None,
        listener: Optional[BaseStreamListener] = None,
        diarization: bool = False,
        events: bool = False,
        mode: str = "batch",
        db_path: str = "sonarwise.db",
        device: Optional[str] = None,
        verbose: bool = False,
        config: Optional[SonarWiseConfig] = None,
        **kwargs,
    ):
        self._config = config or SonarWiseConfig()
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(levelname)s] %(message)s",
            )
            self._config.verbose = True

        self._device = device or self._config.device
        self._mode = mode
        self._diarization_enabled = diarization or self._config.diarization
        self._events_enabled = events or self._config.events

        # Components — lazy loaded
        self._transcriber = transcriber
        self._embedder = embedder
        self._store = store or SQLiteStore(db_path=db_path)
        self._chunker = chunker or VADChunker(
            min_segment_ms=self._config.min_segment_ms,
            max_segment_ms=self._config.max_segment_ms,
        )
        self._diarizer = diarizer
        self._speaker_embedder = speaker_embedder
        self._event_classifier = event_classifier
        self._listener = listener

        # Live mode state
        self._callbacks: Dict[str, List[dict]] = {}
        self._live_thread: Optional[threading.Thread] = None
        self._live_running = False

    # ─── Lazy Component Loading ────────────────────────────────────────

    def _get_transcriber(self) -> BaseTranscriber:
        if self._transcriber is None:
            from sonarwise.core.transcriber import WhisperTranscriber
            self._transcriber = WhisperTranscriber(
                model_size=self._config.transcriber_model,
                device=self._device,
            )
        return self._transcriber

    def _get_embedder(self) -> BaseAudioEmbedder:
        if self._embedder is None:
            from sonarwise.core.embedder import CLAPEmbedder
            self._embedder = CLAPEmbedder(
                model=self._config.embedder_model,
                device=self._device,
            )
        return self._embedder

    def _get_diarizer(self) -> BaseDiarizer:
        if self._diarizer is None:
            if self._diarization_enabled:
                try:
                    from sonarwise.core.diarizer import PyannoteDiarizer
                    self._diarizer = PyannoteDiarizer(device=self._device)
                except ImportError:
                    logger.warning("pyannote not installed, using SimpleDiarizer")
                    self._diarizer = SimpleDiarizer()
            else:
                self._diarizer = SimpleDiarizer()
        return self._diarizer

    def _get_speaker_embedder(self) -> Optional[BaseSpeakerEmbedder]:
        if self._speaker_embedder is None and self._diarization_enabled:
            try:
                from sonarwise.core.speaker_embedder import ECAPAEmbedder
                self._speaker_embedder = ECAPAEmbedder(device=self._device)
            except ImportError:
                logger.warning("speechbrain not installed, speaker embeddings disabled")
                self._speaker_embedder = False  # sentinel: tried and failed
        if self._speaker_embedder is False:
            return None
        return self._speaker_embedder

    def _get_event_classifier(self) -> BaseEventClassifier:
        if self._event_classifier is None:
            if self._events_enabled:
                try:
                    from sonarwise.core.event_classifier import PANNsClassifier
                    self._event_classifier = PANNsClassifier(device=self._device)
                except ImportError:
                    self._event_classifier = SimpleEventClassifier()
            else:
                self._event_classifier = SimpleEventClassifier()
        return self._event_classifier

    # ─── Indexing ──────────────────────────────────────────────────────

    def index(self, filepath: str, skip_existing: bool = False) -> int:
        """Index an audio file. Returns number of segments stored.

        Args:
            filepath: Path to audio file.
            skip_existing: Skip if already indexed.

        Returns:
            Number of segments indexed.
        """
        filepath = os.path.abspath(filepath)

        if skip_existing and filepath in self._store.list_files():
            logger.info(f"Skipping already indexed: {filepath}")
            return 0

        # Remove existing segments for this file (re-index)
        self._store.delete_by_file(filepath)

        logger.info(f"Indexing: {filepath}")

        # Load audio
        audio = AudioIO.load(filepath)
        logger.info(
            f"Loaded: {ms_to_short(audio.duration_ms)} duration, "
            f"{audio.sample_rate}Hz"
        )

        # Chunk
        chunks = self._chunker.chunk(audio)
        logger.info(f"Chunked into {len(chunks)} segments")

        if not chunks:
            logger.warning(f"No segments found in {filepath}")
            return 0

        # Diarize (full file)
        speaker_segments: List[SpeakerSegment] = []
        if self._diarization_enabled:
            speaker_segments = self._get_diarizer().diarize(audio)

        # Process each chunk
        transcriber = self._get_transcriber()
        embedder = self._get_embedder()
        event_clf = self._get_event_classifier() if self._events_enabled else None
        spk_embedder = self._get_speaker_embedder()

        # Get registered speakers for matching
        registered = self._store.get_registered_speakers()

        count = 0
        for chunk in tqdm(chunks, desc="Processing", disable=not self._config.verbose):
            segment = self._process_chunk(
                chunk=chunk,
                filepath=filepath,
                transcriber=transcriber,
                embedder=embedder,
                event_classifier=event_clf,
                speaker_embedder=spk_embedder,
                speaker_segments=speaker_segments,
                registered_speakers=registered,
                source_type="batch",
            )
            self._store.insert(segment)
            count += 1

        logger.info(f"Indexed {count} segments from {filepath}")
        return count

    def index_folder(
        self,
        folder: str,
        recursive: bool = False,
        skip_existing: bool = False,
        formats: Optional[List[str]] = None,
        workers: int = 1,
        on_progress: Optional[Callable] = None,
    ) -> int:
        """Index all audio files in a folder.

        Args:
            folder: Path to folder.
            recursive: Search subdirectories.
            skip_existing: Skip already indexed files.
            formats: File formats to include (default: all supported).
            workers: Not yet implemented — reserved for parallel processing.
            on_progress: Callback for progress updates.

        Returns:
            Total segments indexed.
        """
        allowed = set(formats or SUPPORTED_FORMATS)
        files = []

        if recursive:
            for root, dirs, filenames in os.walk(folder):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower().lstrip(".")
                    if ext in allowed:
                        files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(folder):
                ext = os.path.splitext(fn)[1].lower().lstrip(".")
                if ext in allowed:
                    files.append(os.path.join(folder, fn))

        files.sort()
        total_segments = 0

        for i, fp in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {os.path.basename(fp)}")
            n = self.index(fp, skip_existing=skip_existing)
            total_segments += n
            if on_progress:
                on_progress({"file": fp, "index": i, "total": len(files), "segments": n})

        logger.info(f"Done: {len(files)} files, {total_segments} segments")
        return total_segments

    def _process_chunk(
        self,
        chunk: AudioSeg,
        filepath: str,
        transcriber: BaseTranscriber,
        embedder: BaseAudioEmbedder,
        event_classifier: Optional[BaseEventClassifier],
        speaker_embedder: Optional[BaseSpeakerEmbedder],
        speaker_segments: List[SpeakerSegment],
        registered_speakers: List[dict],
        source_type: str = "batch",
    ) -> Segment:
        """Process a single audio chunk into a Segment."""

        # Transcribe
        transcript = transcriber.transcribe(chunk)

        # Embed audio
        embedding = embedder.embed_audio(chunk)

        # Assign speaker from diarization
        speaker_id = None
        speaker_name = None
        speaker_emb = None

        if speaker_segments:
            speaker_id = self._match_speaker_segment(
                chunk.start_ms, chunk.end_ms, speaker_segments
            )

        # Speaker embedding + name matching
        if speaker_embedder and speaker_id:
            try:
                speaker_emb_np = speaker_embedder.embed(chunk)
                speaker_emb = speaker_emb_np.tolist()

                # Match against registered speakers
                best_name = None
                best_score = 0.0
                for reg in registered_speakers:
                    score = cosine_similarity(
                        speaker_emb_np, np.array(reg["embedding"])
                    )
                    if score > best_score and score > 0.75:
                        best_score = score
                        best_name = reg["name"]
                speaker_name = best_name
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Speaker embedding skipped: {e}")
                speaker_emb = None

        # Event classification
        event_tags = []
        event_confidence = []
        if event_classifier:
            events = event_classifier.classify(chunk)
            event_tags = [e.label for e in events]
            event_confidence = [e.confidence for e in events]

        return Segment(
            filepath=filepath,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            duration_ms=chunk.duration_ms,
            transcript=transcript.text,
            embedding=embedding.tolist(),
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            speaker_embedding=speaker_emb,
            event_tags=event_tags,
            event_confidence=event_confidence,
            language=transcript.language,
            word_timestamps=transcript.words,
            source_type=source_type,
        )

    @staticmethod
    def _match_speaker_segment(
        start_ms: int, end_ms: int, speaker_segments: List[SpeakerSegment]
    ) -> Optional[str]:
        """Find which speaker overlaps most with this time range."""
        best_speaker = None
        best_overlap = 0

        for ss in speaker_segments:
            overlap_start = max(start_ms, ss.start_ms)
            overlap_end = min(end_ms, ss.end_ms)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ss.speaker_id

        return best_speaker

    # ─── Querying ──────────────────────────────────────────────────────

    def query(
        self,
        text: str = "",
        top_k: Optional[int] = None,
        speaker: Optional[str] = None,
        events: Optional[List[str]] = None,
        filepath: Optional[str] = None,
        filepaths: Optional[List[str]] = None,
        language: Optional[str] = None,
        min_confidence: float = 0.0,
        **kwargs,
    ) -> List[SearchResult]:
        """Search indexed audio by text query.

        Args:
            text: Text query.
            top_k: Max results (default from config).
            speaker: Filter by speaker name or ID.
            events: Filter by event types.
            filepath: Filter by single file.
            filepaths: Filter by multiple files.
            language: Filter by language.
            min_confidence: Minimum similarity score.

        Returns:
            List of SearchResult sorted by relevance.
        """
        top_k = top_k or self._config.default_top_k

        # Embed the text query
        query_embedding = self._get_embedder().embed_text(text)

        # Build filters
        filters: Dict[str, Any] = {}
        if speaker:
            # Check if it's a speaker name or ID
            filters["speaker_name"] = speaker
        if filepath:
            filters["filepath"] = os.path.abspath(filepath)
        if language:
            filters["language"] = language
        if events:
            filters["event_tags"] = events

        results = self._store.search(
            embedding=query_embedding,
            top_k=top_k * 3 if filepaths else top_k,
            threshold=min_confidence,
            filters=filters,
        )

        # Filter by multiple filepaths
        if filepaths:
            abs_paths = {os.path.abspath(f) for f in filepaths}
            results = [r for r in results if r.filepath in abs_paths]
            results = results[:top_k]

        return results

    def query_audio(
        self,
        audio_path: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search by audio similarity — find segments that sound similar.

        Args:
            audio_path: Path to query audio clip.
            top_k: Max results.

        Returns:
            List of SearchResult sorted by audio similarity.
        """
        top_k = top_k or self._config.default_top_k

        audio = AudioIO.load(audio_path)
        audio_seg = AudioSeg(
            audio=audio.samples,
            sample_rate=audio.sample_rate,
            start_ms=0,
            end_ms=audio.duration_ms,
        )
        query_embedding = self._get_embedder().embed_audio(audio_seg)

        return self._store.search(
            embedding=query_embedding,
            top_k=top_k,
            filters=kwargs.get("filters"),
        )

    def query_events(
        self,
        event: str,
        top_k: Optional[int] = None,
        filepath: Optional[str] = None,
    ) -> List[EventResult]:
        """Search by event type only.

        Args:
            event: Event label to search for (e.g., "alarm", "machine_fault").
            top_k: Max results.
            filepath: Filter by file.

        Returns:
            List of EventResult.
        """
        top_k = top_k or self._config.default_top_k

        # Get all segments and filter by event
        files = [filepath] if filepath else self._store.list_files()
        results = []

        for fp in files:
            segments = self._store.get_segments_by_file(fp if filepath else fp)
            for seg in segments:
                if event in seg.event_tags:
                    idx = seg.event_tags.index(event)
                    conf = seg.event_confidence[idx] if idx < len(seg.event_confidence) else 0.0
                    results.append(EventResult(
                        filepath=seg.filepath,
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        event_tags=seg.event_tags,
                        event_confidence=seg.event_confidence,
                        transcript=seg.transcript,
                        score=conf,
                    ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─── Speaker Management ────────────────────────────────────────────

    def register_speaker(self, name: str, reference_audio: str) -> None:
        """Register a speaker by name with reference audio.

        Args:
            name: Speaker name.
            reference_audio: Path to reference audio (5-30 seconds of speech).
        """
        embedder = self._get_speaker_embedder()
        if embedder is None:
            raise ImportError(
                "Speaker embedder not available. "
                "Run: pip install sonarwise[speaker]"
            )

        audio = AudioIO.load(reference_audio)
        audio_seg = AudioSeg(
            audio=audio.samples,
            sample_rate=audio.sample_rate,
            start_ms=0,
            end_ms=audio.duration_ms,
        )
        embedding = embedder.embed(audio_seg)

        self._store.register_speaker(
            name=name,
            embedding=embedding.tolist(),
            registered_at=datetime.utcnow().isoformat(),
        )
        logger.info(f"Registered speaker: {name}")

    def list_registered_speakers(self) -> List[RegisteredSpeaker]:
        """List all registered speakers."""
        speakers = self._store.get_registered_speakers()
        return [
            RegisteredSpeaker(
                name=s["name"],
                embedding=s["embedding"],
                registered_at=s["registered_at"],
            )
            for s in speakers
        ]

    def remove_speaker(self, name: str) -> bool:
        """Remove a registered speaker."""
        return self._store.remove_speaker(name)

    def get_speakers(self, filepath: str) -> List[Speaker]:
        """Get all speakers detected in a file."""
        filepath = os.path.abspath(filepath)
        segments = self._store.get_segments_by_file(filepath)

        speaker_map: Dict[str, Speaker] = {}
        for seg in segments:
            sid = seg.speaker_id or "unknown"
            if sid not in speaker_map:
                speaker_map[sid] = Speaker(
                    id=sid, name=seg.speaker_name, duration_ms=0, segment_count=0
                )
            speaker_map[sid].duration_ms += seg.duration_ms
            speaker_map[sid].segment_count += 1
            if seg.speaker_name and not speaker_map[sid].name:
                speaker_map[sid].name = seg.speaker_name

        return sorted(speaker_map.values(), key=lambda s: s.duration_ms, reverse=True)

    def speaker_timeline(self, filepath: str) -> Dict[str, List[tuple]]:
        """Get speaker timeline for a file.

        Returns:
            Dict mapping speaker name/id to list of (start_ms, end_ms) tuples.
        """
        filepath = os.path.abspath(filepath)
        segments = self._store.get_segments_by_file(filepath)

        timeline: Dict[str, List[tuple]] = {}
        for seg in segments:
            key = seg.speaker_name or seg.speaker_id or "unknown"
            if key not in timeline:
                timeline[key] = []
            timeline[key].append((seg.start_ms, seg.end_ms))

        return timeline

    def speaker_stats(self, filepath: str) -> Dict[str, SpeakerStats]:
        """Get detailed speaker statistics for a file."""
        filepath = os.path.abspath(filepath)
        segments = self._store.get_segments_by_file(filepath)

        total_duration = sum(s.duration_ms for s in segments)
        stats_map: Dict[str, dict] = {}

        for seg in segments:
            sid = seg.speaker_name or seg.speaker_id or "unknown"
            if sid not in stats_map:
                stats_map[sid] = {
                    "speaker_id": seg.speaker_id or "unknown",
                    "speaker_name": seg.speaker_name,
                    "total_duration_ms": 0,
                    "segment_count": 0,
                    "longest_segment_ms": 0,
                    "word_count": 0,
                }
            stats = stats_map[sid]
            stats["total_duration_ms"] += seg.duration_ms
            stats["segment_count"] += 1
            stats["longest_segment_ms"] = max(stats["longest_segment_ms"], seg.duration_ms)
            stats["word_count"] += len(seg.transcript.split())

        result = {}
        for sid, stats in stats_map.items():
            pct = (stats["total_duration_ms"] / total_duration * 100) if total_duration else 0
            avg = (
                stats["total_duration_ms"] / stats["segment_count"]
                if stats["segment_count"]
                else 0
            )
            result[sid] = SpeakerStats(
                speaker_id=stats["speaker_id"],
                speaker_name=stats["speaker_name"],
                total_duration_ms=stats["total_duration_ms"],
                percentage=round(pct, 1),
                segment_count=stats["segment_count"],
                avg_segment_ms=round(avg, 1),
                longest_segment_ms=stats["longest_segment_ms"],
                word_count=stats["word_count"],
            )

        return result

    def find_speaker_across(
        self, speaker: str, folders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Find a registered speaker across all indexed files.

        Args:
            speaker: Speaker name.
            folders: Optional folder filter.

        Returns:
            Dict with files_found_in, total_segments, total_duration_ms, breakdown.
        """
        all_files = self._store.list_files()
        if folders:
            abs_folders = [os.path.abspath(f) for f in folders]
            all_files = [
                fp for fp in all_files
                if any(fp.startswith(folder) for folder in abs_folders)
            ]

        breakdown = []
        total_segments = 0
        total_duration = 0

        for fp in all_files:
            segments = self._store.get_segments_by_file(fp)
            matching = [
                s for s in segments
                if s.speaker_name == speaker or s.speaker_id == speaker
            ]
            if matching:
                dur = sum(s.duration_ms for s in matching)
                breakdown.append({
                    "file": fp,
                    "segments": len(matching),
                    "duration_ms": dur,
                })
                total_segments += len(matching)
                total_duration += dur

        return {
            "speaker": speaker,
            "files_found_in": len(breakdown),
            "total_segments": total_segments,
            "total_duration_ms": total_duration,
            "breakdown": breakdown,
        }

    def interaction_matrix(self, filepath: str) -> Dict[tuple, int]:
        """Get speaker interaction matrix — who speaks after whom.

        Returns:
            Dict mapping (speaker_a, speaker_b) to count of transitions.
        """
        filepath = os.path.abspath(filepath)
        segments = self._store.get_segments_by_file(filepath)

        matrix: Dict[tuple, int] = {}
        for i in range(1, len(segments)):
            prev = segments[i - 1].speaker_name or segments[i - 1].speaker_id or "unknown"
            curr = segments[i].speaker_name or segments[i].speaker_id or "unknown"
            if prev != curr:
                key = (prev, curr)
                matrix[key] = matrix.get(key, 0) + 1

        return matrix

    # ─── Live Mode ─────────────────────────────────────────────────────

    def on(self, event_type: str, **kwargs) -> Callable:
        """Register a callback for live mode events.

        Usage:
            @sw.on("transcript")
            def on_speech(segment):
                print(segment.transcript)

            @sw.on("keyword", words=["budget", "risk"])
            def on_keyword(segment):
                send_alert(segment.transcript)
        """
        def decorator(func: Callable) -> Callable:
            if event_type not in self._callbacks:
                self._callbacks[event_type] = []
            self._callbacks[event_type].append({
                "func": func,
                "kwargs": kwargs,
            })
            return func
        return decorator

    def listen(self, source: str = "microphone", **kwargs) -> None:
        """Start live audio listening.

        Args:
            source: "microphone", "system_audio", or URL (rtsp://, ws://).
        """
        if self._listener is None:
            if source == "microphone":
                from sonarwise.core.stream_listener import MicListener
                self._listener = MicListener(
                    sample_rate=self._config.sample_rate,
                    chunk_duration_ms=self._config.chunk_duration_ms,
                )
            elif source.startswith(("rtsp://", "ws://", "wss://")):
                raise NotImplementedError(
                    f"Stream source '{source}' not yet implemented. "
                    "Use a custom BaseStreamListener."
                )
            else:
                # Try as file path for testing
                from sonarwise.core.stream_listener import FileStreamListener
                self._listener = FileStreamListener(
                    filepath=source,
                    chunk_duration_ms=self._config.chunk_duration_ms,
                )

        self._listener.start()
        self._live_running = True

        # Run processing in background thread
        self._live_thread = threading.Thread(
            target=self._live_loop, daemon=True
        )
        self._live_thread.start()
        logger.info(f"Live mode started: {source}")

    def _live_loop(self):
        """Background loop for live audio processing."""
        transcriber = self._get_transcriber()
        embedder = self._get_embedder()
        event_clf = self._get_event_classifier() if self._events_enabled else None
        spk_embedder = self._get_speaker_embedder()
        registered = self._store.get_registered_speakers()

        prev_speaker = None

        for chunk in self._listener.stream():
            if not self._live_running:
                break

            audio_seg = AudioSeg(
                audio=chunk.data,
                sample_rate=chunk.sample_rate,
                start_ms=chunk.timestamp_ms,
                end_ms=chunk.timestamp_ms + self._config.chunk_duration_ms,
            )

            segment = self._process_chunk(
                chunk=audio_seg,
                filepath="__live__",
                transcriber=transcriber,
                embedder=embedder,
                event_classifier=event_clf,
                speaker_embedder=spk_embedder,
                speaker_segments=[],
                registered_speakers=registered,
                source_type="live",
            )

            # Store live segment
            self._store.insert(segment)

            # Fire callbacks
            self._fire_callbacks(segment, prev_speaker)
            prev_speaker = segment.speaker_id

    def _fire_callbacks(self, segment: Segment, prev_speaker: Optional[str]):
        """Fire registered callbacks based on segment data."""
        result = SearchResult(
            segment_id=segment.segment_id,
            filepath=segment.filepath,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            duration_ms=segment.duration_ms,
            transcript=segment.transcript,
            speaker_id=segment.speaker_id,
            speaker_name=segment.speaker_name,
            event_tags=segment.event_tags,
            event_confidence=segment.event_confidence,
            language=segment.language,
            source_type="live",
        )

        # "transcript" callbacks
        for cb in self._callbacks.get("transcript", []):
            try:
                cb["func"](result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # "keyword" callbacks
        for cb in self._callbacks.get("keyword", []):
            words = cb["kwargs"].get("words", [])
            transcript_lower = segment.transcript.lower()
            if any(w.lower() in transcript_lower for w in words):
                try:
                    cb["func"](result)
                except Exception as e:
                    logger.error(f"Keyword callback error: {e}")

        # "speaker_change" callbacks
        if prev_speaker and segment.speaker_id != prev_speaker:
            for cb in self._callbacks.get("speaker_change", []):
                try:
                    cb["func"](type("Event", (), {
                        "old_speaker": prev_speaker,
                        "new_speaker": segment.speaker_id,
                        "timestamp_ms": segment.start_ms,
                    })())
                except Exception as e:
                    logger.error(f"Speaker change callback error: {e}")

        # "sound_event" callbacks
        for cb in self._callbacks.get("sound_event", []):
            target_events = cb["kwargs"].get("events", [])
            if any(e in segment.event_tags for e in target_events):
                try:
                    cb["func"](type("Event", (), {
                        "event_tags": segment.event_tags,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "filepath": segment.filepath,
                    })())
                except Exception as e:
                    logger.error(f"Event callback error: {e}")

        # "silence" callbacks
        for cb in self._callbacks.get("silence", []):
            duration_ms = cb["kwargs"].get("duration_ms", 10000)
            if "silence" in segment.event_tags and segment.duration_ms >= duration_ms:
                try:
                    cb["func"](result)
                except Exception as e:
                    logger.error(f"Silence callback error: {e}")

        # "custom" callbacks
        for cb in self._callbacks.get("custom", []):
            condition = cb["kwargs"].get("condition")
            if condition and condition(result):
                try:
                    cb["func"](result)
                except Exception as e:
                    logger.error(f"Custom callback error: {e}")

    def stop(self):
        """Stop live listening."""
        self._live_running = False
        if self._listener:
            self._listener.stop()
        if self._live_thread:
            self._live_thread.join(timeout=5)
        logger.info("Live mode stopped")

    # ─── Management ────────────────────────────────────────────────────

    def list_sources(self) -> List[Source]:
        """List all indexed audio sources."""
        files = self._store.list_files()
        sources = []
        for fp in files:
            segments = self._store.get_segments_by_file(fp)
            total_dur = sum(s.duration_ms for s in segments)
            speakers = len(set(s.speaker_id for s in segments if s.speaker_id))
            indexed_at = segments[0].indexed_at if segments else ""
            sources.append(Source(
                filepath=fp,
                segment_count=len(segments),
                duration_ms=total_dur,
                indexed_at=indexed_at,
                speakers=speakers,
            ))
        return sources

    def remove(self, filepath: str) -> int:
        """Remove a file from the index. Returns segments deleted."""
        return self._store.delete_by_file(os.path.abspath(filepath))

    def remove_all(self) -> int:
        """Remove all indexed data."""
        total = 0
        for fp in self._store.list_files():
            total += self._store.delete_by_file(fp)
        return total

    def stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        files = self._store.list_files()
        total_segments = self._store.count()
        total_duration = 0
        all_speakers = set()
        all_languages = set()
        event_dist: Dict[str, int] = {}

        for fp in files:
            segments = self._store.get_segments_by_file(fp)
            for seg in segments:
                total_duration += seg.duration_ms
                if seg.speaker_id:
                    all_speakers.add(seg.speaker_id)
                all_languages.add(seg.language)
                for evt in seg.event_tags:
                    event_dist[evt] = event_dist.get(evt, 0) + 1

        registered = self._store.get_registered_speakers()
        db_size = 0
        if hasattr(self._store, "db_path") and os.path.exists(self._store.db_path):
            db_size = round(os.path.getsize(self._store.db_path) / (1024 * 1024), 2)

        return {
            "total_files": len(files),
            "total_segments": total_segments,
            "total_duration_ms": total_duration,
            "total_speakers": len(all_speakers),
            "registered_speakers": len(registered),
            "db_size_mb": db_size,
            "unique_languages": sorted(all_languages),
            "event_distribution": event_dist,
        }

    # ─── Export ─────────────────────────────────────────────────────────

    def export(
        self,
        filepath: str,
        format: str = "json",
        output: Optional[str] = None,
    ) -> str:
        """Export indexed data to file.

        Args:
            filepath: Source audio filepath to export data for.
            format: Export format — "json", "csv", "srt", "vtt", "txt", "notes".
            output: Output file path (auto-generated if None).

        Returns:
            Path to exported file.
        """
        filepath = os.path.abspath(filepath)
        segments = self._store.get_segments_by_file(filepath)

        if not segments:
            raise ValueError(f"No segments found for: {filepath}")

        base = os.path.splitext(os.path.basename(filepath))[0]
        if output is None:
            output = f"{base}.{format}"

        exporters = {
            "json": self._export_json,
            "csv": self._export_csv,
            "srt": self._export_srt,
            "vtt": self._export_vtt,
            "txt": self._export_txt,
            "notes": self._export_notes,
        }

        exporter = exporters.get(format)
        if exporter is None:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters)}")

        exporter(segments, output, filepath)
        logger.info(f"Exported to {output}")
        return output

    @staticmethod
    def _export_json(segments: List[Segment], output: str, filepath: str):
        data = []
        for seg in segments:
            data.append({
                "segment_id": seg.segment_id,
                "filepath": seg.filepath,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "duration_ms": seg.duration_ms,
                "transcript": seg.transcript,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "event_tags": seg.event_tags,
                "language": seg.language,
                "indexed_at": seg.indexed_at,
            })
        with open(output, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _export_csv(segments: List[Segment], output: str, filepath: str):
        headers = [
            "segment_id", "start_ms", "end_ms", "duration_ms",
            "transcript", "speaker_id", "speaker_name", "event_tags", "language",
        ]
        with open(output, "w") as f:
            f.write(",".join(headers) + "\n")
            for seg in segments:
                row = [
                    seg.segment_id,
                    str(seg.start_ms),
                    str(seg.end_ms),
                    str(seg.duration_ms),
                    f'"{seg.transcript}"',
                    seg.speaker_id or "",
                    seg.speaker_name or "",
                    "|".join(seg.event_tags),
                    seg.language,
                ]
                f.write(",".join(row) + "\n")

    @staticmethod
    def _export_srt(segments: List[Segment], output: str, filepath: str):
        with open(output, "w") as f:
            for i, seg in enumerate(segments, 1):
                start = ms_to_srt_timestamp(seg.start_ms)
                end = ms_to_srt_timestamp(seg.end_ms)
                speaker = f"[{seg.speaker_name or seg.speaker_id or 'Unknown'}] "
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{speaker}{seg.transcript}\n\n")

    @staticmethod
    def _export_vtt(segments: List[Segment], output: str, filepath: str):
        with open(output, "w") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = ms_to_vtt_timestamp(seg.start_ms)
                end = ms_to_vtt_timestamp(seg.end_ms)
                speaker = f"<v {seg.speaker_name or seg.speaker_id or 'Unknown'}>"
                f.write(f"{start} --> {end}\n")
                f.write(f"{speaker}{seg.transcript}\n\n")

    @staticmethod
    def _export_txt(segments: List[Segment], output: str, filepath: str):
        with open(output, "w") as f:
            for seg in segments:
                ts = ms_to_short(seg.start_ms)
                speaker = seg.speaker_name or seg.speaker_id or "Unknown"
                f.write(f"[{ts}] {speaker}: {seg.transcript}\n")

    @staticmethod
    def _export_notes(segments: List[Segment], output: str, filepath: str):
        total_dur = sum(s.duration_ms for s in segments)
        speakers = {}
        for seg in segments:
            key = seg.speaker_name or seg.speaker_id or "Unknown"
            if key not in speakers:
                speakers[key] = {"duration_ms": 0, "word_count": 0}
            speakers[key]["duration_ms"] += seg.duration_ms
            speakers[key]["word_count"] += len(seg.transcript.split())

        with open(output, "w") as f:
            f.write(f"## Meeting: {os.path.basename(filepath)}\n")
            f.write(
                f"Duration: {ms_to_short(total_dur)} | "
                f"Speakers: {len(speakers)}\n\n"
            )

            f.write("### Speaker Summary\n")
            for name, info in sorted(
                speakers.items(), key=lambda x: x[1]["duration_ms"], reverse=True
            ):
                pct = round(info["duration_ms"] / total_dur * 100, 1) if total_dur else 0
                f.write(f"- {name} ({pct}%): {info['word_count']} words\n")

            f.write("\n### Transcript\n")
            for seg in segments:
                ts = ms_to_short(seg.start_ms)
                speaker = seg.speaker_name or seg.speaker_id or "Unknown"
                f.write(f"[{ts}] **{speaker}**: {seg.transcript}\n")

    # ─── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Clean up all resources."""
        self.stop() if self._live_running else None
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (
            f"SonarWise(db='{self._store.db_path if hasattr(self._store, 'db_path') else '?'}', "
            f"diarization={self._diarization_enabled}, "
            f"events={self._events_enabled}, "
            f"mode='{self._mode}')"
        )
