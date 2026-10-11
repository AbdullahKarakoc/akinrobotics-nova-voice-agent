"""Backend-agnostic streaming chat interface.

Messages use the OpenAI chat format internally (role/content/tool_calls/tool_call_id);
each backend converts to its own wire format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCalls:
    calls: list[ToolCall]


@dataclass
class LLMStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_ms: float | None = None  # prefill time, if the server reports it
    eval_ms: float | None = None  # decode time, if the server reports it
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float | None:
        if self.eval_ms and self.completion_tokens:
            return self.completion_tokens / (self.eval_ms / 1000)
        return None


@dataclass
class Done:
    stats: LLMStats


LLMEvent = TextDelta | ToolCalls | Done


class ChatLLM(Protocol):
    model: str

    def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMEvent]: ...

    async def aclose(self) -> None: ...
