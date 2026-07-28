"""sonarwise — Example 4: Export Formats

Export indexed audio data to SRT, VTT, JSON, CSV, TXT, and meeting notes.

Usage:
    python examples/04_export.py path/to/audio.wav
"""

import sys
from sonarwise import SonarWise


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/04_export.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    sw = SonarWise(diarization=True, events=True, verbose=True)

    # Index
    print(f"\nIndexing: {audio_path}")
    sw.index(audio_path)

    # Export all formats
    formats = {
        "json": "Structured data (full segment info)",
        "csv": "Spreadsheet-friendly table",
        "srt": "Subtitles for video players",
        "vtt": "Subtitles for web players",
        "txt": "Plain text transcript with timestamps",
        "notes": "Meeting notes with speaker summary",
    }

    print("\nExporting all formats:")
    for fmt, description in formats.items():
        output = sw.export(audio_path, format=fmt)
        print(f"  {fmt:5s} -> {output:30s} ({description})")

    sw.close()
    print("\nDone. Check the exported files.")


if __name__ == "__main__":
    main()
