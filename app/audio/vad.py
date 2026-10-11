"""Streaming voice activity detection and end-of-turn detection.

`SileroVAD` wraps the Silero VAD v6 ONNX model (bundled with faster-whisper,
~2 MB, < 1 ms per 32 ms frame on CPU). `Endpointer` turns the per-frame speech
probabilities into "user started speaking" / "user finished speaking" events
with hysteresis, a minimum speech duration and a pre-roll buffer.
"""

from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz, the window size Silero expects
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
_CONTEXT_SAMPLES = 64


def _default_model_path() -> str:
    import faster_whisper

    return os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")


class SileroVAD:
    """Shared ONNX session; create one `VADStream` per audio stream."""

    def __init__(self, model_path: str | None = None):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            model_path or _default_model_path(), providers=["CPUExecutionProvider"], sess_options=opts
        )

    def stream(self) -> VADStream:
        return VADStream(self.session)


class VADStream:
    """Keeps the recurrent state and audio context of one stream."""

    def __init__(self, session: ort.InferenceSession):
        self._session = session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability of one 512-sample float32 frame."""
        x = np.concatenate([self._context, frame])[None, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-_CONTEXT_SAMPLES:]
        return float(prob.reshape(-1)[0])


class VadEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VadEvent:
    type: VadEventType
    audio: np.ndarray | None = None  # float32 utterance, only on SPEECH_END
    speech_end_time: float = 0.0  # perf_counter() when the last voiced frame arrived


@dataclass
class EndpointerConfig:
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 200
    speech_pad_ms: int = 300
    max_utterance_s: float = 25.0


@dataclass
class Endpointer:
    """Frame-level state machine: idle -> (onset) -> speaking -> (silence) -> idle."""

    vad: VADStream
    config: EndpointerConfig = field(default_factory=EndpointerConfig)

    def __post_init__(self) -> None:
        self.min_speech_ms = self.config.min_speech_ms
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, self.config.speech_pad_ms // FRAME_MS)
        )
        self._reset_turn()

    @property
    def neg_threshold(self) -> float:
        return max(self.config.threshold - 0.15, 0.05)

    @property
    def in_speech(self) -> bool:
        return self._triggered

    def _reset_turn(self) -> None:
        self._triggered = False
        self._onset_frames: list[np.ndarray] = []
        self._speech_frames: list[np.ndarray] = []
        self._voiced_ms = 0
        self._silence_ms = 0
        self._last_voiced_time = 0.0
        self._last_voiced_index = 0

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._reset_turn()

    def process(self, pcm16: bytes) -> list[VadEvent]:
        """Feed raw 16 kHz mono int16 audio of any length; returns events."""
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        events: list[VadEvent] = []
        while len(self._pending) >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            event = self._process_frame(frame)
            if event:
                events.append(event)
        return events

    def _process_frame(self, frame: np.ndarray) -> VadEvent | None:
        prob = self.vad(frame)
        now = time.perf_counter()

        if not self._triggered:
            if prob >= self.config.threshold:
                self._onset_frames.append(frame)
                self._voiced_ms += FRAME_MS
                if self._voiced_ms >= self.min_speech_ms:
                    self._triggered = True
                    self._speech_frames = list(self._preroll) + self._onset_frames
                    self._onset_frames = []
                    self._last_voiced_time = now
                    self._last_voiced_index = len(self._speech_frames)
                    self._preroll.clear()
                    return VadEvent(VadEventType.SPEECH_START)
            else:
                # Not enough consecutive speech: the blip goes into the pre-roll.
                for f in self._onset_frames:
                    self._preroll.append(f)
                self._onset_frames = []
                self._voiced_ms = 0
                self._preroll.append(frame)
            return None

        self._speech_frames.append(frame)
        if prob >= self.neg_threshold:
            self._silence_ms = 0
            if prob >= self.config.threshold:
                self._last_voiced_time = now
                self._last_voiced_index = len(self._speech_frames)
        else:
            self._silence_ms += FRAME_MS

        duration_s = len(self._speech_frames) * FRAME_MS / 1000
        if self._silence_ms >= self.config.min_silence_ms or duration_s >= self.config.max_utterance_s:
            keep = min(len(self._speech_frames), self._last_voiced_index + self.config.speech_pad_ms // FRAME_MS)
            audio = np.concatenate(self._speech_frames[:keep])
            event = VadEvent(VadEventType.SPEECH_END, audio=audio, speech_end_time=self._last_voiced_time)
            self._reset_turn()
            return event
        return None
