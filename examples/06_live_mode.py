"""sonarwise — Example 6: Live Mode

Real-time audio listening with keyword and event callbacks.
Uses a file as simulated live source (replace with "microphone" for real use).

Usage:
    python examples/06_live_mode.py path/to/audio.wav
"""

import sys
import time

from sonarwise import SonarWise
from sonarwise.core.stream_listener import FileStreamListener
from sonarwise.utils.time_utils import ms_to_short


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/06_live_mode.py <audio_file>")
        print("\nFor real microphone: change source to 'microphone'")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Use file stream for demo (replace with "microphone" for real use)
    sw = SonarWise(
        mode="live",
        events=True,
        verbose=True,
        listener=FileStreamListener(filepath=audio_path, chunk_duration_ms=3000),
    )

    # Register callbacks
    @sw.on("transcript")
    def on_speech(segment):
        ts = ms_to_short(segment.start_ms)
        speaker = segment.speaker_name or segment.speaker_id or ""
        prefix = f"[{speaker}] " if speaker else ""
        print(f"  [{ts}] {prefix}{segment.transcript}")

    @sw.on("keyword", words=["budget", "deadline", "risk", "blocker"])
    def on_keyword(segment):
        print(f"  >>> KEYWORD ALERT: {segment.transcript}")

    @sw.on("sound_event", events=["alarm", "glass_break", "machine_fault"])
    def on_danger(event):
        print(f"  >>> EVENT ALERT: {event.event_tags}")

    # Start listening
    print(f"\nListening to: {audio_path}")
    print("(Ctrl+C to stop)\n")

    sw.listen(source=audio_path)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sw.stop()
        print("\n\nStopped. Searching recorded audio...")

        # Everything heard is now searchable
        results = sw.query("budget", top_k=3)
        if results:
            print("\nSearch results from live session:")
            for r in results:
                print(f"  {r.transcript} (score: {r.score})")

    sw.close()


if __name__ == "__main__":
    main()
