"""Per-turn latency metrics and running aggregates (served at /api/metrics)."""

from __future__ import annotations

import collections
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnMetrics:
    turn_id: int
    mode: str  # "voice" | "text"
    language: str = ""
    # input side
    audio_s: float | None = None  # length of the user's utterance
    endpoint_ms: float | None = None  # VAD: last voiced frame -> end-of-turn decision
    stt_ms: float | None = None
    stt_rtf: float | None = None
    # reasoning
    retrieval_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_prompt_tokens: int | None = None
    llm_prefill_ms: float | None = None
    llm_completion_tokens: int | None = None
    llm_tokens_per_s: float | None = None
    llm_rounds: int | None = None
    tool_ms: float | None = None
    # output side
    tts_first_ms: float | None = None  # synthesis time of the first sentence
    tts_rtf: float | None = None
    tts_audio_s: float | None = None
    # the numbers that matter to the user
    first_audio_ms: float | None = None  # user stopped speaking -> first agent audio sent
    total_ms: float | None = None
    interrupted: bool = False
    tools: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items() if v is not None}


SUMMARY_FIELDS = (
    "first_audio_ms", "endpoint_ms", "stt_ms", "stt_rtf", "retrieval_ms", "llm_ttft_ms",
    "llm_prefill_ms", "llm_tokens_per_s", "tool_ms", "tts_first_ms", "tts_rtf", "total_ms",
)  # fmt: skip


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


class MetricsStore:
    def __init__(self, maxlen: int = 1000):
        self.turns: collections.deque[TurnMetrics] = collections.deque(maxlen=maxlen)

    def add(self, metrics: TurnMetrics) -> None:
        self.turns.append(metrics)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"turns": len(self.turns)}
        for mode in ("voice", "text"):
            turns = [t for t in self.turns if t.mode == mode and not t.interrupted]
            if not turns:
                continue
            stats = {}
            for name in SUMMARY_FIELDS:
                values = [getattr(t, name) for t in turns if getattr(t, name) is not None]
                if values:
                    stats[name] = {
                        "n": len(values),
                        "mean": round(statistics.fmean(values), 3),
                        "p50": round(_percentile(values, 0.5), 3),
                        "p90": round(_percentile(values, 0.9), 3),
                    }
            out[mode] = stats
        return out
