<p align="center">
  <img src="https://raw.githubusercontent.com/VK-Ant/sonarwise/main/assets/sonarwise_ai.png" alt="sonarwise" width="800"/>
</p>

<h1 align="center">sonarwise</h1>
<p align="center"><b>Pluggable audio perception engine. Hear. Search. Retrieve.</b></p>

<p align="center">
  <a href="https://pypi.org/project/sonarwise"><img src="https://img.shields.io/pypi/v/sonarwise?color=blue" alt="PyPI"/></a>
  <a href="https://github.com/VK-Ant/sonarwise/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9+-green" alt="Python"></a>
</p>

---

**sonarwise** indexes any audio like meetings, calls, podcasts, factory floors, lectures and makes it searchable by text, speaker, sound events, or audio similarity. Every component is pluggable: swap transcription, embedding, diarization, or storage without changing your code.

## Features

- **Transcription** : Whisper, Faster Whisper, or bring your own ASR
- **Audio Embeddings** : CLAP joint text-audio space for semantic search
- **Speaker Diarization** : know who said what (pyannote)
- **Speaker Registry** : track speakers across files by voiceprint
- **Audio Event Detection** : detect alarms, machinery, glass breaks, and 50+ sound types
- **Live Streaming** : real-time transcription with keyword and event callbacks
- **Export** : SRT, VTT, JSON, CSV, TXT, meeting notes
- **Pluggable Architecture** : every component swappable via base classes
- **CLI** : full command-line interface

## Install

```bash
# Core (no ML dependencies)
pip install sonarwise

# With all features
pip install sonarwise[all]

# Pick what you need
pip install sonarwise[whisper]          # Whisper transcription
pip install sonarwise[faster-whisper]   # Faster Whisper (CTranslate2)
pip install sonarwise[clap]            # CLAP audio embeddings
pip install sonarwise[diarization]     # Speaker diarization
pip install sonarwise[speaker]         # Speaker identification
pip install sonarwise[events]          # Audio event detection
pip install sonarwise[live]            # Live microphone capture
```

**Requires:** ffmpeg (`sudo apt install ffmpeg` or `brew install ffmpeg`)

## Quick Start

```python
from sonarwise import SonarWise

sw = SonarWise(diarization=True, events=True)

# Index audio
sw.index("meeting.wav")
sw.index_folder("./recordings/")

# Search by text
results = sw.query("budget discussion", top_k=5)
for r in results:
    print(f"[{r.speaker_name}] {r.transcript} (score: {r.score})")

# Search by audio similarity
results = sw.query_audio("alarm_clip.wav", top_k=5)

# Search by speaker
results = sw.query("budget", speaker="Ant", top_k=5)

# Search by event
results = sw.query_events(event="machine_fault", top_k=5)
```

## Speaker Intelligence

```python
# Register a speaker
sw.register_speaker("Ant", reference_audio="ant_voice.wav")

# Query by speaker
results = sw.query("budget", speaker="Ant")

# Speaker timeline
timeline = sw.speaker_timeline("meeting.wav")

# Speaker stats
stats = sw.speaker_stats("meeting.wav")

# Find speaker across files
presence = sw.find_speaker_across(speaker="Ant", folders=["./meetings/"])
```

## Live Mode

```python
sw = SonarWise(mode="live", diarization=True, events=True)

@sw.on("transcript")
def on_speech(segment):
    print(f"[{segment.speaker_name}] {segment.transcript}")

@sw.on("keyword", words=["budget", "deadline", "risk"])
def on_keyword(segment):
    send_alert(segment.transcript)

@sw.on("sound_event", events=["alarm", "glass_break"])
def on_danger(event):
    trigger_alert(event)

sw.listen(source="microphone")
```

## Plug Any Model

Every component is swappable:

```python
from sonarwise import SonarWise
from sonarwise.core.transcriber import BaseTranscriber

class MyTranscriber(BaseTranscriber):
    def transcribe(self, audio):
        return my_model.process(audio)

sw = SonarWise(transcriber=MyTranscriber())
```

**Pluggable slots:**

| Component | Base Class | Default |
|-----------|-----------|---------|
| Transcriber | `BaseTranscriber` | Whisper |
| Audio Embedder | `BaseAudioEmbedder` | CLAP |
| Vector Store | `BaseVectorStore` | SQLite |
| Chunker | `BaseChunker` | Silero VAD |
| Diarizer | `BaseDiarizer` | pyannote |
| Speaker Embedder | `BaseSpeakerEmbedder` | ECAPA-TDNN |
| Event Classifier | `BaseEventClassifier` | PANNs |
| Stream Listener | `BaseStreamListener` | sounddevice |

## CLI

```bash
sonarwise index meeting.wav
sonarwise query "budget discussion"
sonarwise speakers meeting.wav
sonarwise timeline meeting.wav
sonarwise listen --source microphone --on-keyword "budget,risk"
sonarwise export meeting.wav --format srt
sonarwise stats
```

## Export

```python
sw.export("meeting.wav", format="srt", output="subtitles.srt")
sw.export("meeting.wav", format="json", output="segments.json")
sw.export("meeting.wav", format="txt", output="transcript.txt")
sw.export("meeting.wav", format="notes", output="meeting_notes.md")
```

## Ecosystem

sonarwise is part of the VK-Ant AI perception ecosystem:

| Library | Purpose | Tagline |
|---------|---------|---------|
| [SightRAG](https://github.com/VK-Ant/SightRAG) | Visual perception | See. Search. Retrieve. |
| **sonarwise** | Audio perception | Hear. Search. Retrieve. |
| [adaptive-intelligence](https://github.com/VK-Ant/adaptive-intelligence) | Reasoning & memory | Learn. Remember. Adapt. |
| [llmevalkit](https://github.com/VK-Ant/llmevalkit) | Evaluation | Evaluate. Score. Improve. |

```python
from sightrag import SightRAG
from sonarwise import SonarWise
from adaptive_intelligence import AdaptiveRAG

brain = AdaptiveRAG()
brain.register_source("visual", SightRAG())
brain.register_source("audio", SonarWise())

# One query hits both eyes and ears
results = brain.query("when did machine 3 start failing?")
```
## License

Apache 2.0

## Author

Built by **Venkatkumar Rajan**

- GitHub: https://github.com/VK-Ant
- LinkedIn: https://linkedin.com/in/vk-ant
- Portfolio: https://vk-ant.github.io/Venkatkumar
