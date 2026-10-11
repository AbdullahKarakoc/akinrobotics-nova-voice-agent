import numpy as np

from app.audio.vad import FRAME_MS, FRAME_SAMPLES, Endpointer, EndpointerConfig, VadEventType


class ScriptedVAD:
    """Returns pre-defined speech probabilities, one per frame."""

    def __init__(self, probs):
        self.probs = list(probs)

    def __call__(self, frame):
        return self.probs.pop(0) if self.probs else 0.0

    def reset(self):
        pass


def frames(n: int) -> bytes:
    return np.zeros(n * FRAME_SAMPLES, dtype=np.int16).tobytes()


def run(probs, **cfg):
    config = EndpointerConfig(min_speech_ms=96, min_silence_ms=320, speech_pad_ms=64, **cfg)
    ep = Endpointer(ScriptedVAD(probs), config)
    return ep.process(frames(len(probs)))


def test_speech_then_silence_produces_start_and_end():
    probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 12
    events = run(probs)
    assert [e.type for e in events] == [VadEventType.SPEECH_START, VadEventType.SPEECH_END]
    audio_ms = len(events[1].audio) / 16
    # speech (20 frames) + pre-roll (2 frames) + trailing pad (2 frames), trailing silence trimmed
    assert abs(audio_ms - 24 * FRAME_MS) <= FRAME_MS


def test_short_blips_are_ignored():
    probs = [0.9, 0.9, 0.0, 0.0, 0.9, 0.0] + [0.0] * 20
    assert run(probs) == []


def test_brief_pauses_do_not_end_the_turn():
    probs = [0.9] * 10 + [0.0] * 6 + [0.9] * 10 + [0.0] * 12  # 192 ms pause < 320 ms
    events = run(probs)
    assert [e.type for e in events] == [VadEventType.SPEECH_START, VadEventType.SPEECH_END]


def test_partial_frames_are_buffered():
    ep = Endpointer(ScriptedVAD([0.9] * 10), EndpointerConfig(min_speech_ms=96))
    data = frames(10)
    events = []
    for i in range(0, len(data), 300):  # odd-sized network packets
        events += ep.process(data[i : i + 300])
    assert [e.type for e in events] == [VadEventType.SPEECH_START]
