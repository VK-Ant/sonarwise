"""sonarwise — Example 3: Audio Event Detection

Detect non-speech sounds: alarms, machinery, glass breaks, etc.

Usage:
    python examples/03_event_detection.py path/to/audio.wav
"""

import sys
from sonarwise import SonarWise
from sonarwise.utils.time_utils import ms_to_short


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/03_event_detection.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Enable event detection
    sw = SonarWise(events=True, verbose=True)

    # Index
    print(f"\nIndexing with event detection: {audio_path}")
    sw.index(audio_path)

    # Show all detected events
    print("\nEvent distribution:")
    stats = sw.stats()
    for event, count in sorted(stats["event_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {event}: {count}")

    # Query specific events
    for event_type in ["alarm", "machine_fault", "speech", "silence"]:
        results = sw.query_events(event=event_type, top_k=3)
        if results:
            print(f"\n'{event_type}' events:")
            for r in results:
                ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
                conf = r.event_confidence[r.event_tags.index(event_type)] if event_type in r.event_tags else 0
                print(f"  [{ts}] confidence: {conf}")

    sw.close()


if __name__ == "__main__":
    main()
