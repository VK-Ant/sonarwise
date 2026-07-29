"""
sonarwise — RTSP Audio Device Demo
For IP cameras, security systems, and audio-enabled devices.

This demo shows how to use sonarwise with:
- RTSP audio streams (IP cameras like Axis, Hikvision, Dahua)
- PipeWire / PulseAudio system audio
- USB microphones
- Any audio input device

Usage:
    python demo_rtsp.py rtsp://192.168.1.100/audio
    python demo_rtsp.py microphone
    python demo_rtsp.py path/to/recorded_audio.wav
"""

import sys
import os
import time
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from sonarwise import SonarWise
from sonarwise.core.stream_listener import BaseStreamListener, FileStreamListener
from sonarwise.utils.time_utils import ms_to_short


def divider(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}\n")


# ─── Use Case 1: Index pre-recorded device audio ──────────

def demo_batch(audio_path):
    """Index a recorded audio file from any device."""

    divider("Use Case 1: Batch Index (recorded audio)")

    sw = SonarWise(events=True, db_path="demo_device.db", verbose=True)

    print(f"  Indexing: {audio_path}")
    count = sw.index(audio_path)
    print(f"  Indexed {count} segments\n")

    # Search
    print("  Searching for 'alarm':")
    results = sw.query("alarm", top_k=3)
    for r in results:
        ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
        print(f"    [{ts}] {r.transcript} (score: {r.score})")

    # Event detection
    print("\n  Sound events detected:")
    stats = sw.stats()
    for event, count in stats.get("event_distribution", {}).items():
        print(f"    {event}: {count}")

    sw.close()


# ─── Use Case 2: Simulate live RTSP stream ────────────────

def demo_live_from_file(audio_path):
    """Simulate RTSP-like live streaming using a recorded file."""

    divider("Use Case 2: Live Stream (simulated from file)")

    sw = SonarWise(
        mode="live",
        events=True,
        db_path="demo_live.db",
        verbose=True,
        listener=FileStreamListener(filepath=audio_path, chunk_duration_ms=3000),
    )

    segments_received = []

    @sw.on("transcript")
    def on_speech(segment):
        ts = ms_to_short(segment.start_ms)
        print(f"  [{ts}] {segment.transcript}")
        segments_received.append(segment)

    @sw.on("keyword", words=["alarm", "help", "emergency", "fault", "error"])
    def on_keyword(segment):
        print(f"  >>> ALERT: {segment.transcript}")

    print(f"  Streaming from: {audio_path}")
    print("  (simulating real-time RTSP input)\n")

    sw.listen(source=audio_path)

    # Wait for stream to finish
    time.sleep(5)
    sw.stop()

    print(f"\n  Received {len(segments_received)} segments in real-time")
    print("  All segments are now searchable in the database")

    # Query the live data
    results = sw.query("hello", top_k=3)
    if results:
        print(f"\n  Post-stream search for 'hello':")
        for r in results:
            print(f"    {r.transcript} (score: {r.score})")

    sw.close()


# ─── Use Case 3: RTSP connection guide ─────────────────────

def print_rtsp_guide():
    """Show how to connect to real RTSP devices."""

    divider("Use Case 3: Connecting Real RTSP Devices")

    print("""  sonarwise supports RTSP audio via custom listeners.

  Example: Axis Camera (AUDIO_IN)
  ───────────────────────────────
  RTSP URL format:
    rtsp://username:password@192.168.1.100/axis-media/media.amp

  With sonarwise + ffmpeg listener:

    from sonarwise import SonarWise
    from sonarwise.core.stream_listener import BaseStreamListener
    import subprocess
    import numpy as np

    class RTSPListener(BaseStreamListener):
        def __init__(self, url, sample_rate=16000, chunk_ms=2000):
            self.url = url
            self.sr = sample_rate
            self.chunk_ms = chunk_ms
            self._process = None
            self._running = False

        def start(self):
            self._process = subprocess.Popen(
                [
                    "ffmpeg", "-i", self.url,
                    "-f", "f32le", "-acodec", "pcm_f32le",
                    "-ar", str(self.sr), "-ac", "1",
                    "-loglevel", "error", "pipe:1",
                ],
                stdout=subprocess.PIPE,
            )
            self._running = True

        def stop(self):
            self._running = False
            if self._process:
                self._process.terminate()

        def stream(self):
            chunk_bytes = int(self.sr * self.chunk_ms / 1000) * 4
            elapsed = 0
            while self._running:
                data = self._process.stdout.read(chunk_bytes)
                if not data:
                    break
                samples = np.frombuffer(data, dtype=np.float32)
                from sonarwise.core.stream_listener import AudioChunk
                yield AudioChunk(data=samples, sample_rate=self.sr, timestamp_ms=elapsed)
                elapsed += self.chunk_ms

    # Usage:
    sw = SonarWise(
        mode="live",
        events=True,
        listener=RTSPListener("rtsp://admin:pass@192.168.1.100/audio")
    )

    @sw.on("sound_event", events=["alarm", "glass_break"])
    def on_alert(event):
        send_notification(event)

    sw.listen()

  ───────────────────────────────
  Supported devices:
    - Axis cameras (AUDIO_IN via RTSP)
    - Hikvision DVR/NVR
    - Dahua cameras
    - Any RTSP-capable audio source
    - PipeWire / PulseAudio (Linux system audio)
    - USB microphones (via sounddevice)
  """)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo_rtsp.py <audio_file>     (batch + live demo)")
        print("  python demo_rtsp.py --guide           (RTSP connection guide)")
        sys.exit(1)

    # Cleanup old dbs
    for db in ["demo_device.db", "demo_live.db"]:
        if os.path.exists(db):
            os.remove(db)

    if sys.argv[1] == "--guide":
        print_rtsp_guide()
    else:
        audio_path = sys.argv[1]
        demo_batch(audio_path)
        demo_live_from_file(audio_path)
        print_rtsp_guide()

        # Cleanup
        for f in ["demo_device.db", "demo_live.db"]:
            if os.path.exists(f):
                os.remove(f)

    divider("Done!")


if __name__ == "__main__":
    main()
