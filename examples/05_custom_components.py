"""sonarwise — Example 5: Pluggable Custom Components

Bring your own transcriber, embedder, or any component.
Every slot is swappable with one class + one method.

Usage:
    python examples/05_custom_components.py path/to/audio.wav
"""

import sys
import numpy as np

from sonarwise import SonarWise
from sonarwise.core.transcriber import BaseTranscriber
from sonarwise.core.embedder import BaseAudioEmbedder
from sonarwise.core.event_classifier import BaseEventClassifier
from sonarwise.core.models import AudioSegment, TranscriptResult, EventTag
from sonarwise.utils.time_utils import ms_to_short


# ─── Custom Transcriber ─────────────────────────────────────────
class MyTranscriber(BaseTranscriber):
    """Example: wrap any ASR model or API."""

    def transcribe(self, audio: AudioSegment) -> TranscriptResult:
        # Replace this with your model:
        # - A fine-tuned Whisper for your language
        # - A cloud API (Deepgram, AssemblyAI, Azure)
        # - A custom CTC/Conformer model
        return TranscriptResult(
            text="your transcription here",
            language="en",
            confidence=0.95,
        )


# ─── Custom Embedder ────────────────────────────────────────────
class MyEmbedder(BaseAudioEmbedder):
    """Example: wrap any audio embedding model."""

    def embed_audio(self, audio: AudioSegment) -> np.ndarray:
        # Replace with your model:
        # - BEATs, PANNs encoder, Audio-MAE
        # - Any model that outputs a fixed-size vector
        return np.random.randn(512).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        # Text embedding for cross-modal search
        # Use CLAP, or embed transcript with sentence-transformers
        return np.random.randn(512).astype(np.float32)


# ─── Custom Event Classifier ────────────────────────────────────
class MyFactoryDetector(BaseEventClassifier):
    """Example: custom detector trained on your factory sounds."""

    def classify(self, audio: AudioSegment) -> list:
        # Replace with your custom model trained on:
        # - Your specific machinery sounds
        # - Your factory alarm types
        # - Your environment
        rms = float(np.sqrt(np.mean(audio.audio ** 2)))
        if rms > 0.5:
            return [EventTag(label="compressor_fault", confidence=0.92)]
        return [EventTag(label="normal_operation", confidence=0.85)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/05_custom_components.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Plug in your custom components
    sw = SonarWise(
        transcriber=MyTranscriber(),
        embedder=MyEmbedder(),
        event_classifier=MyFactoryDetector(),
        events=True,
        verbose=True,
    )

    # Same API — nothing else changes
    print(f"\nIndexing with custom components: {audio_path}")
    count = sw.index(audio_path)
    print(f"Indexed {count} segments")

    results = sw.query("your query", top_k=3)
    print(f"\nResults: {len(results)}")
    for r in results:
        ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
        print(f"  [{ts}] {r.transcript} events={r.event_tags}")

    sw.close()


if __name__ == "__main__":
    main()
