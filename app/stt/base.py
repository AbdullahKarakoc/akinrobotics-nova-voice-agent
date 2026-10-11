from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float
    audio_s: float
    processing_s: float

    @property
    def rtf(self) -> float:
        return self.processing_s / self.audio_s if self.audio_s else 0.0


class SpeechToText(Protocol):
    """Any STT engine: 16 kHz mono float32 audio in, text + detected language out."""

    def transcribe(self, audio: np.ndarray, languages: list[str]) -> Transcript: ...
