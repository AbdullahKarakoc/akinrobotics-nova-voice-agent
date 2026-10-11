from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SpeechAudio:
    pcm16: bytes  # mono, little-endian int16
    sample_rate: int
    processing_s: float

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.processing_s / self.duration_s if self.duration_s else 0.0


class TextToSpeech(Protocol):
    """Any TTS engine: one sentence of text in, PCM audio out."""

    def synthesize(self, text: str, language: str) -> SpeechAudio: ...
