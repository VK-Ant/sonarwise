"""Audio event classification — detect non-speech sounds."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from sonarwise.core.models import AudioSegment, EventTag

logger = logging.getLogger(__name__)

# Standard event taxonomy
EVENT_CATEGORIES = {
    "industrial": [
        "machine_fault", "grinding", "drilling", "hammering",
        "motor_running", "compressor", "vibration", "impact",
        "metal_clang", "steam_release", "conveyor_belt",
    ],
    "alarm": [
        "alarm", "siren", "beep", "buzzer", "fire_alarm", "smoke_detector",
    ],
    "environment": [
        "glass_break", "door_slam", "door_knock", "footsteps",
        "rain", "wind", "thunder", "traffic", "car_horn", "construction",
    ],
    "human": [
        "speech", "shout", "whisper", "crying", "coughing", "sneezing",
        "laughter", "clapping", "breathing_heavy", "crowd_noise",
    ],
    "animal": [
        "dog_bark", "cat_meow", "bird_chirp", "rooster", "insect_buzz",
    ],
    "media": [
        "music", "musical_instrument", "singing", "applause",
        "ringtone", "notification_sound",
    ],
    "silence": [
        "silence", "background_noise", "white_noise",
    ],
}

# Flat list of all events
ALL_EVENTS = [evt for cat in EVENT_CATEGORIES.values() for evt in cat]


class BaseEventClassifier(ABC):
    """Base class for event classifiers. Implement classify() to bring your own model."""

    @abstractmethod
    def classify(self, audio: AudioSegment) -> List[EventTag]:
        """Classify audio events in a segment.

        Args:
            audio: AudioSegment to classify.

        Returns:
            List of EventTag(label, confidence).
        """
        raise NotImplementedError


class PANNsClassifier(BaseEventClassifier):
    """Pretrained Audio Neural Networks (PANNs) classifier — CNN14.

    Install: pip install sonarwise[events]
    """

    def __init__(
        self,
        model: str = "CNN14",
        device: Optional[str] = None,
        min_confidence: float = 0.3,
        top_k: int = 5,
    ):
        self.model_name = model
        self.device = device
        self.min_confidence = min_confidence
        self.top_k = top_k
        self._model = None
        self._labels = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchaudio

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            bundle = torchaudio.pipelines.VGGISH
            self._model = bundle.get_model().to(device)
            self._model.eval()
            self._sample_rate = bundle.sample_rate

            logger.info(f"PANNs/VGGish event classifier loaded on {device}")
        except (ImportError, Exception) as e:
            logger.warning(
                f"Event classifier not available ({e}). "
                "Install: pip install sonarwise[events]"
            )
            self._model = None

    def classify(self, audio: AudioSegment) -> List[EventTag]:
        # Energy-based fallback if model not available
        if self._model is None:
            self._load_model()
        if self._model is None:
            return self._energy_classify(audio)

        return self._model_classify(audio)

    def _model_classify(self, audio: AudioSegment) -> List[EventTag]:
        import torch
        import torchaudio

        waveform = torch.FloatTensor(audio.audio).unsqueeze(0)

        # Resample if needed
        if audio.sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio.sample_rate,
                new_freq=self._sample_rate,
            )
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            output = self._model(waveform)
            probs = torch.sigmoid(output).cpu().numpy().flatten()

        events = []
        for idx in np.argsort(probs)[::-1][: self.top_k]:
            conf = float(probs[idx])
            if conf >= self.min_confidence:
                label = f"event_{idx}"
                events.append(EventTag(label=label, confidence=round(conf, 3)))

        if not events:
            events.append(EventTag(label="unknown", confidence=0.5))

        return events

    @staticmethod
    def _energy_classify(audio: AudioSegment) -> List[EventTag]:
        """Simple energy-based classification fallback."""
        rms = float(np.sqrt(np.mean(audio.audio**2)))
        if rms < 0.005:
            return [EventTag(label="silence", confidence=0.95)]
        elif rms < 0.05:
            return [EventTag(label="background_noise", confidence=0.7)]
        else:
            return [EventTag(label="speech", confidence=0.6)]


class SimpleEventClassifier(BaseEventClassifier):
    """Simple energy + zero-crossing based classifier. No ML dependencies."""

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def classify(self, audio: AudioSegment) -> List[EventTag]:
        samples = audio.audio
        rms = float(np.sqrt(np.mean(samples**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))))

        events = []

        if rms < 0.005:
            events.append(EventTag(label="silence", confidence=0.95))
        elif rms < 0.02:
            events.append(EventTag(label="background_noise", confidence=0.7))
        elif zcr > 0.5 and rms > 0.1:
            events.append(EventTag(label="speech", confidence=0.6))
        elif rms > 0.3:
            events.append(EventTag(label="loud_event", confidence=0.7))
        else:
            events.append(EventTag(label="speech", confidence=0.5))

        return [e for e in events if e.confidence >= self.min_confidence]
