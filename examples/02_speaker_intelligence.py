"""sonarwise — Example 2: Speaker Intelligence

Index with diarization, register speakers, query by speaker.

Usage:
    python examples/02_speaker_intelligence.py path/to/audio.wav
"""

import sys
from sonarwise import SonarWise
from sonarwise.utils.time_utils import ms_to_short


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/02_speaker_intelligence.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Enable diarization
    sw = SonarWise(diarization=True, verbose=True)

    # Index
    print(f"\nIndexing with diarization: {audio_path}")
    sw.index(audio_path)

    # Show speakers
    print("\nSpeakers detected:")
    speakers = sw.get_speakers(audio_path)
    for s in speakers:
        name = s.name or s.id
        dur = ms_to_short(s.duration_ms)
        print(f"  {name}: {dur} ({s.segment_count} segments)")

    # Show timeline
    print("\nSpeaker timeline:")
    timeline = sw.speaker_timeline(audio_path)
    for speaker, ranges in timeline.items():
        parts = [f"{ms_to_short(s)}-{ms_to_short(e)}" for s, e in ranges[:5]]
        suffix = " ..." if len(ranges) > 5 else ""
        print(f"  {speaker}: {' | '.join(parts)}{suffix}")

    # Show stats
    print("\nSpeaker stats:")
    stats = sw.speaker_stats(audio_path)
    for name, s in stats.items():
        print(f"  {name}: {s.percentage}% talk time, {s.word_count} words")

    # Query by speaker
    print("\nQuery 'budget' for speaker_0:")
    results = sw.query("budget", speaker="speaker_0", top_k=3)
    for r in results:
        ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
        print(f"  [{ts}] {r.transcript} (score: {r.score})")

    sw.close()


if __name__ == "__main__":
    main()
