"""sonarwise — Quick 10-second test script.

Run: python test_quick.py examples/sample_audio.m4a
"""

import sys
import os
import logging

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from sonarwise import SonarWise
from sonarwise.utils.time_utils import ms_to_short

if len(sys.argv) < 2:
    print("Usage: python test_quick.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

# Clean old db
if os.path.exists("sonarwise.db"):
    os.remove("sonarwise.db")

print("=" * 50)
print("  sonarwise v0.1.0 — Quick Test")
print("  Hear. Search. Retrieve.")
print("=" * 50)

sw = SonarWise(verbose=True)

# 1. Index
print(f"\n[1] Indexing: {audio_path}")
count = sw.index(audio_path)
print(f"    Result: {count} segments indexed")

# 2. Query
print("\n[2] Text query: 'hello'")
results = sw.query("hello", top_k=3)
for r in results:
    ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
    print(f"    [{ts}] {r.transcript} (score: {r.score})")

# 3. Stats
print("\n[3] Stats:")
stats = sw.stats()
print(f"    Files: {stats['total_files']}")
print(f"    Segments: {stats['total_segments']}")
print(f"    Duration: {ms_to_short(stats['total_duration_ms'])}")

# 4. Export
print("\n[4] Export SRT:")
srt_path = sw.export(audio_path, format="srt", output="test_output.srt")
with open(srt_path) as f:
    print(f"    {f.read().strip()}")

# 5. Cleanup
sw.close()
os.remove("sonarwise.db")
os.remove("test_output.srt")

print("\n" + "=" * 50)
print("  ALL PASSED — sonarwise is working!")
print("=" * 50)
