"""sonarwise — Example 1: Basic Index and Query

Index an audio file and search it by text.

Usage:
    python examples/01_basic_usage.py path/to/audio.wav
"""

import sys
from sonarwise import SonarWise
from sonarwise.utils.time_utils import ms_to_short


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/01_basic_usage.py <audio_file>")
        print("Example: python examples/01_basic_usage.py meeting.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Initialize — core features only, no diarization or events
    sw = SonarWise(verbose=True)

    # Index the audio file
    print(f"\nIndexing: {audio_path}")
    count = sw.index(audio_path)
    print(f"Indexed {count} segments\n")

    # Interactive query loop
    print("Search your audio (type 'quit' to exit):\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break

        results = sw.query(query, top_k=5)
        if not results:
            print("  No results found.\n")
            continue

        for i, r in enumerate(results, 1):
            ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
            print(f"  {i}. [{ts}] {r.transcript} (score: {r.score})")
        print()

    sw.close()
    print("Done.")


if __name__ == "__main__":
    main()
