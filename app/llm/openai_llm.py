"""OpenAI-compatible /chat/completions backend.

Works with vLLM, llama.cpp `llama-server`, LM Studio, LocalAI, Ollama's /v1,
as well as hosted APIs (OpenAI, Groq, Together, ...). `base_url` must include
the version prefix, e.g. http://localhost:8080/v1.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import Done, LLMEvent, LLMStats, TextDelta, ToolCall, ToolCalls


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 200,
        timeout_s: float = 90.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(timeout_s, connect=5.0)
        )

    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The OpenAI format wants tool-call arguments as a JSON string."""
        wire = []
        for m in messages:
            if m.get("tool_calls"):
                m = {
                    **m,
                    "tool_calls": [
                        {**c, "function": {**c["function"], "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str) else json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
                        for c in m["tool_calls"]
                    ],
                }
            wire.append(m)
        return wire

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools

        partial: dict[int, dict[str, str]] = {}  # tool calls arrive as fragments keyed by index
        stats = LLMStats()
        async with self.client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"LLM error {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    stats.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    stats.completion_tokens = chunk["usage"].get("completion_tokens", 0)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield TextDelta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = partial.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""

        if partial:
            calls = []
            for i, slot in sorted(partial.items()):
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"], arguments=args))
            yield ToolCalls(calls)
        yield Done(stats)

    async def aclose(self) -> None:
        await self.client.aclose()
