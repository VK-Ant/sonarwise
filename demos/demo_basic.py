"""
sonarwise — Basic Python Demo
Hear. Search. Retrieve.

A step-by-step walkthrough of sonarwise features.
Works with any audio file (wav, mp3, m4a, flac, ogg).

Usage:
    python demo_basic.py path/to/your_audio.m4a
"""

import sys
import os
import logging

# Clean logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from sonarwise import SonarWise, __version__
from sonarwise.utils.time_utils import ms_to_short


def divider(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_basic.py <audio_file>")
        print("Example: python demo_basic.py meeting.m4a")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Clean previous runs
    if os.path.exists("sonarwise_demo.db"):
        os.remove("sonarwise_demo.db")

    divider(f"sonarwise v{__version__} — Basic Demo")
    print("  Pluggable Audio Perception Engine")
    print("  Hear. Search. Retrieve.\n")

    # ────────────────────────────────────────────
    # STEP 1: Initialize
    # ────────────────────────────────────────────
    divider("Step 1: Initialize sonarwise")

    sw = SonarWise(db_path="sonarwise_demo.db", verbose=True)
    print("  SonarWise initialized with default components:")
    print("  - Transcriber: Whisper (base)")
    print("  - Embedder: CLAP (text-audio joint space)")
    print("  - Store: SQLite")
    print("  - Chunker: Silero VAD")

    # ────────────────────────────────────────────
    # STEP 2: Index audio
    # ────────────────────────────────────────────
    divider("Step 2: Index audio file")

    print(f"  Input: {audio_path}\n")
    count = sw.index(audio_path)
    print(f"\n  Result: {count} segments indexed")

    # ────────────────────────────────────────────
    # STEP 3: Search by text
    # ────────────────────────────────────────────
    divider("Step 3: Search by text query")

    queries = ["hello", "thank", "opportunity"]
    for q in queries:
        results = sw.query(q, top_k=3)
        print(f"  Query: \"{q}\"")
        if results:
            for r in results:
                ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
                print(f"    [{ts}] {r.transcript} (score: {r.score})")
        else:
            print(f"    No results found.")
        print()

    # ────────────────────────────────────────────
    # STEP 4: View stats
    # ────────────────────────────────────────────
    divider("Step 4: Index statistics")

    stats = sw.stats()
    print(f"  Files indexed:    {stats['total_files']}")
    print(f"  Total segments:   {stats['total_segments']}")
    print(f"  Total duration:   {ms_to_short(stats['total_duration_ms'])}")
    print(f"  Languages:        {', '.join(stats['unique_languages'])}")
    print(f"  Database size:    {stats['db_size_mb']} MB")

    # ────────────────────────────────────────────
    # STEP 5: Export
    # ────────────────────────────────────────────
    divider("Step 5: Export to SRT subtitles")

    srt_path = sw.export(audio_path, format="srt", output="demo_output.srt")
    with open(srt_path) as f:
        content = f.read()
    print(f"  Exported to: {srt_path}\n")
    print(f"  Content:\n{content}")

    # ────────────────────────────────────────────
    # STEP 6: Export to JSON
    # ────────────────────────────────────────────
    divider("Step 6: Export to JSON")

    json_path = sw.export(audio_path, format="json", output="demo_output.json")
    print(f"  Exported to: {json_path}")

    # ────────────────────────────────────────────
    # Cleanup
    # ────────────────────────────────────────────
    sw.close()

    divider("Demo Complete!")
    print("  sonarwise is working end-to-end.")
    print("  pip install sonarwise")
    print("  GitHub: github.com/VK-Ant/sonarwise\n")


if __name__ == "__main__":
    main()
