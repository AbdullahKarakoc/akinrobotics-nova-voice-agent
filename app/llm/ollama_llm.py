"""Ollama native /api/chat backend (streams tokens and tool calls, reports prefill/decode timings)."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import Done, LLMEvent, LLMStats, TextDelta, ToolCall, ToolCalls

log = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 200,
        num_ctx: int = 4096,
        num_thread: int | None = None,
        keep_alive: str = "30m",
        timeout_s: float = 90.0,
        think: bool | None = None,
    ):
        self.model = model
        self.think = think  # only sent when set: non-reasoning models reject the field
        self.options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
        if num_thread:
            self.options["num_thread"] = num_thread
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0))

    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        names_by_id: dict[str, str] = {}
        wire = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    names_by_id[c["id"]] = c["function"]["name"]
                    args = c["function"]["arguments"]
                    calls.append({"function": {"name": c["function"]["name"], "arguments": json.loads(args) if isinstance(args, str) else args}})
                wire.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            elif m["role"] == "tool":
                wire.append({"role": "tool", "content": m["content"], "tool_name": names_by_id.get(m.get("tool_call_id", ""), "")})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if tools:
            body["tools"] = tools
        if self.think is not None:
            body["think"] = self.think

        calls: list[ToolCall] = []
        async with self.client.stream("POST", "/api/chat", json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"Ollama error {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                message = chunk.get("message") or {}
                if message.get("content"):
                    yield TextDelta(message["content"])
                for c in message.get("tool_calls") or []:
                    fn = c.get("function", {})
                    calls.append(ToolCall(id=c.get("id") or uuid.uuid4().hex[:8], name=fn.get("name", ""), arguments=fn.get("arguments") or {}))
                if chunk.get("done"):
                    if calls:
                        yield ToolCalls(calls)
                    yield Done(
                        LLMStats(
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                            prompt_eval_ms=chunk.get("prompt_eval_duration", 0) / 1e6,
                            eval_ms=chunk.get("eval_duration", 0) / 1e6,
                            extra={"load_ms": chunk.get("load_duration", 0) / 1e6},
                        )
                    )
                    return

    async def has_model(self) -> bool:
        return (await self.client.post("/api/show", json={"model": self.model})).status_code != 404

    async def pull_model(self) -> None:
        """Pull the model into the server (e.g. a host Ollama), logging progress every 10%."""
        logged: dict[str, int] = {}
        async with self.client.stream("POST", "/api/pull", json={"model": self.model}, timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama pull error: {chunk['error']}")
                total, done = chunk.get("total") or 0, chunk.get("completed") or 0
                if total < 50_000_000:  # only the weights layer is worth reporting
                    continue
                percent = done * 100 // total
                digest = chunk.get("digest", "")
                if percent // 10 > logged.get(digest, -1):
                    logged[digest] = percent // 10
                    log.info(
                        "llm download %d%% (%d / %d MB)", percent, done // 1_000_000, total // 1_000_000,
                        extra={"data": {"model": self.model}},
                    )

    async def aclose(self) -> None:
        await self.client.aclose()
