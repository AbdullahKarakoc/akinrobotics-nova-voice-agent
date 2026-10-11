"""The conversational agent: retrieval -> LLM (streaming, with tool calls) -> text stream.

The agent is transport-agnostic: it consumes text and yields text deltas, so
the same code serves the voice pipeline, the text WebSocket and the HTTP API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from app.agent.prompts import ERROR_REPLIES, TOOL_FILLERS, build_system_prompt, build_user_message
from app.agent.speech_text import ungrounded_numbers
from app.config import Settings
from app.knowledge import Hit, Retriever
from app.llm import ChatLLM, Done, LLMStats, TextDelta, ToolCall, ToolCalls
from app.tools import ToolRegistry, ToolResult

log = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """Everything that happened while answering one user message (for UI, logs and metrics)."""

    answer: str = ""  # what the model said (kept in history)
    spoken: str = ""  # answer plus any filler phrases
    hits: list[Hit] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_stats: list[LLMStats] = field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ttft_ms: float | None = None  # first answer token, measured from LLM request start
    tool_ms: float = 0.0
    total_ms: float = 0.0
    ungrounded: list[str] = field(default_factory=list)
    error: str | None = None
    # Everything the answer may legitimately draw numbers from (question, knowledge, tool output).
    # Grows during the turn, so the voice pipeline can check each sentence before speaking it.
    evidence: list[str] = field(default_factory=list)

    def evidence_text(self) -> str:
        return " ".join(self.evidence)

    def sources(self) -> list[dict[str, Any]]:
        docs = [{"type": "document", "id": h.chunk.id, "title": h.chunk.title, "score": round(h.score, 3)} for h in self.hits]
        tools = [{"type": "tool", "id": r.call.name, "args": r.call.arguments, "ok": r.ok} for r in self.tool_results]
        return docs + tools


class LeakedToolCallGuard:
    """Small models sometimes write a tool call as plain text (`{"name": ..., "arguments": ...}` or
    `<tool_call>...</tool_call>`) instead of using the function-calling channel. That must never
    be spoken. A reply that *starts* like JSON/markup is held back until the round ends and then
    either turned into real tool calls or released as ordinary text."""

    def __init__(self, tool_names: Iterable[str]):
        self.names = set(tool_names)
        self.held = ""
        self.holding: bool | None = None  # undecided until the first non-blank character

    def feed(self, text: str) -> str:
        """Returns the part of `text` that is safe to speak now."""
        if self.holding is False:
            return text
        self.held += text
        stripped = self.held.lstrip()
        if not stripped:
            return ""
        if self.holding is None:
            self.holding = stripped[0] in "{[,<`"
        if self.holding:
            return ""
        released, self.held = self.held, ""
        return released

    def finish(self) -> tuple[list[ToolCall], str]:
        """(recovered tool calls, text to release) at the end of an LLM round."""
        if not self.holding:
            return [], ""
        decoder, calls = json.JSONDecoder(), []
        for i, ch in enumerate(self.held):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(self.held[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("name") in self.names:
                args = obj.get("arguments") or obj.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                calls.append(ToolCall(id=f"text_{len(calls)}", name=obj["name"], arguments=args))
        return (calls, "") if calls else ([], self.held)


class Conversation:
    """Rolling chat history. Retrieved context is not stored: it is re-fetched per turn."""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.messages: list[dict[str, Any]] = []

    def add(self, user: str, assistant: str) -> None:
        self.messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.messages = self.messages[-2 * self.max_turns :]

    def last_user_message(self) -> str | None:
        return next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), None)

    def clear(self) -> None:
        self.messages.clear()


class Agent:
    def __init__(self, llm: ChatLLM, retriever: Retriever, tools: ToolRegistry, settings: Settings):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools
        self.settings = settings
        self.full_knowledge = settings.knowledge_mode == "full"
        self.system_prompt = build_system_prompt(retriever.chunks if self.full_knowledge else None)
        self.knowledge_text = " ".join(c.full_text for c in retriever.chunks)
        self.chunk_labels = {c.id: f"K{i}" for i, c in enumerate(retriever.chunks, 1)}

    async def warmup(self) -> None:
        """Load the model and prime the server's prompt cache with the static system prompt + tools."""
        start = time.perf_counter()
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "Merhaba"}]
        async for _ in self.llm.stream_chat(messages, self.tools.schemas()):
            pass
        log.info("llm warm", extra={"data": {"model": self.llm.model, "ms": round((time.perf_counter() - start) * 1000)}})

    def _retrieve(self, question: str, conversation: Conversation) -> list[Hit]:
        s = self.settings
        hits = self.retriever.search(question, s.rag_top_k, s.rag_min_score)
        previous = conversation.last_user_message()
        if previous and (not hits or len(question.split()) <= 6):
            # Follow-ups like "and its battery?" need the previous question to be searchable.
            merged = {h.chunk.id: h for h in self.retriever.search(f"{previous} {question}", s.rag_top_k, s.rag_min_score)}
            for h in hits:
                merged.setdefault(h.chunk.id, h)
            hits = sorted(merged.values(), key=lambda h: -h.score)[: s.rag_top_k]
        # Every context token costs prefill time: drop chunks far below the best match.
        return [h for h in hits if h.score >= hits[0].score - s.rag_relative_cut] if hits else []

    async def respond(self, question: str, language: str, conversation: Conversation, turn: AgentTurn) -> AsyncIterator[str]:
        """Yield the answer as text deltas; fills `turn` with sources and metrics."""
        start = time.perf_counter()
        routed = self.tools.route(question)
        retrieval = asyncio.create_task(asyncio.to_thread(self._retrieve, question, conversation))
        if routed:
            turn.tool_results = list(await asyncio.gather(*(self.tools.execute(c) for c in routed)))
            turn.tool_ms = (time.perf_counter() - start) * 1000
        turn.hits = await retrieval
        turn.retrieval_ms = (time.perf_counter() - start) * 1000
        knowledge = [self.knowledge_text] if self.full_knowledge else [h.chunk.full_text for h in turn.hits]
        turn.evidence = [question, *knowledge, *(r.content for r in turn.tool_results)]

        if self.full_knowledge:
            relevant = [self.chunk_labels[h.chunk.id] for h in turn.hits]
            user_message = build_user_message(question, language, None, turn.tool_results, relevant)
        else:
            user_message = build_user_message(question, language, turn.hits, turn.tool_results)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *conversation.messages,
            {"role": "user", "content": user_message},
        ]
        tool_schemas = self.tools.schemas()
        llm_start = time.perf_counter()
        try:
            for round_no in range(self.settings.max_tool_rounds + 1):
                allow_tools = round_no < self.settings.max_tool_rounds
                round_text, calls = "", []
                guard = LeakedToolCallGuard(self.tools.tools)
                async for event in self.llm.stream_chat(messages, tool_schemas if allow_tools else None):
                    if isinstance(event, TextDelta):
                        if turn.llm_ttft_ms is None:
                            turn.llm_ttft_ms = (time.perf_counter() - llm_start) * 1000
                        if text := guard.feed(event.text):
                            round_text += text
                            turn.answer += text
                            turn.spoken += text
                            yield text
                    elif isinstance(event, ToolCalls):
                        calls = event.calls
                    elif isinstance(event, Done):
                        turn.llm_stats.append(event.stats)
                leaked, released = guard.finish()
                if released:
                    round_text += released
                    turn.answer += released
                    turn.spoken += released
                    yield released
                if leaked and not calls and allow_tools:
                    log.info("recovered tool call written as text", extra={"data": {"calls": [c.name for c in leaked]}})
                    calls = leaked
                if not calls:
                    break

                if not turn.spoken.strip():
                    filler = TOOL_FILLERS.get(language, TOOL_FILLERS["tr"])
                    turn.spoken += filler + " "
                    yield filler + " "
                tool_start = time.perf_counter()
                results = await asyncio.gather(*(self.tools.execute(c) for c in calls))
                turn.tool_ms += (time.perf_counter() - tool_start) * 1000
                turn.tool_results.extend(results)
                turn.evidence.extend(r.content for r in results)
                messages.append(
                    {
                        "role": "assistant",
                        "content": round_text,
                        "tool_calls": [
                            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                            for c in calls
                        ],
                    }
                )
                messages += [{"role": "tool", "tool_call_id": r.call.id, "content": r.content} for r in results]
        except Exception as exc:
            log.exception("llm failed")
            turn.error = repr(exc)
            reply = ERROR_REPLIES.get(language, ERROR_REPLIES["tr"])
            turn.spoken += reply
            yield reply

        turn.ungrounded = ungrounded_numbers(turn.answer, turn.evidence_text())
        if turn.ungrounded:
            log.warning("answer contains numbers not found in sources", extra={"data": {"numbers": turn.ungrounded}})
        turn.total_ms = (time.perf_counter() - start) * 1000
        if turn.answer.strip():
            conversation.add(question, turn.answer.strip())
