"""Tool (function-calling) registry.

A tool is a plain Python function plus a JSON-schema description. Adding a new
data source = writing a function and decorating it:

    @registry.tool("get_x", "What it returns and when to use it.", {"id": {"type": "string"}})
    def get_x(id: str) -> dict: ...

Handlers may be sync (run in a worker thread) or async. They return a JSON-
serialisable dict; failures are caught and reported to the LLM as an error
payload so the model can tell the user instead of inventing a result.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.llm.base import ToolCall

log = logging.getLogger(__name__)


Router = Callable[[str], "dict[str, Any] | None"]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    handler: Callable[..., Any]
    # Optional deterministic trigger: returns arguments when the user text unambiguously asks
    # for this tool (e.g. contains an order number). Routed tools run before the LLM, in
    # parallel with retrieval, which saves a whole LLM round and doesn't depend on a small
    # model choosing to call the tool.
    router: Router | None = None
    # False: only routers call it; its schema is not offered to the LLM (no prompt tokens, and
    # the model cannot decide to use it, e.g. for an off-topic question).
    llm_visible: bool = True

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters, "required": self.required},
            },
        }


@dataclass
class ToolResult:
    call: ToolCall
    content: str  # JSON string handed back to the LLM
    ok: bool
    ms: float


class ToolRegistry:
    def __init__(self, timeout_s: float = 8.0):
        self.tools: dict[str, Tool] = {}
        self.timeout_s = timeout_s
        self._closers: list[Callable[[], None]] = []

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def on_close(self, fn: Callable[[], None]) -> None:
        """Register cleanup for a tool source that holds a connection (e.g. an MCP session)."""
        self._closers.append(fn)

    def close(self) -> None:
        for fn in self._closers:
            fn()
        self._closers.clear()

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        required: list[str] | None = None,
        router: Router | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            params = parameters or {}
            required_params = required if required is not None else list(params)
            self.add(Tool(name, description, params, required_params, fn, router))
            return fn

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values() if t.llm_visible]

    def route(self, text: str) -> list[ToolCall]:
        """Tool calls whose router fires on this user text."""
        calls = []
        for tool in self.tools.values():
            if tool.router and (args := tool.router(text)) is not None:
                calls.append(ToolCall(id=f"route_{tool.name}", name=tool.name, arguments=args))
        return calls

    async def execute(self, call: ToolCall) -> ToolResult:
        start = time.perf_counter()
        tool = self.tools.get(call.name)
        try:
            if tool is None:
                raise KeyError(f"unknown tool '{call.name}'")
            params = inspect.signature(tool.handler).parameters
            # Generic (**kwargs) handlers, e.g. MCP proxies, accept what the tool's schema declares.
            var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
            accepted = set(tool.parameters) if var_kw else set(params)
            kwargs = {k: v for k, v in call.arguments.items() if k in accepted}
            if inspect.iscoroutinefunction(tool.handler):
                result = await asyncio.wait_for(tool.handler(**kwargs), self.timeout_s)
            else:
                result = await asyncio.wait_for(asyncio.to_thread(tool.handler, **kwargs), self.timeout_s)
            ok = not (isinstance(result, dict) and "error" in result)
        except Exception as exc:  # reported to the model, never raised into the conversation loop
            log.warning("tool failed", extra={"data": {"tool": call.name, "error": repr(exc)}})
            result, ok = {"error": f"Tool failed: {exc}. Tell the user the information is unavailable right now."}, False
        ms = (time.perf_counter() - start) * 1000
        log.info("tool call", extra={"data": {"tool": call.name, "args": call.arguments, "ok": ok, "ms": round(ms)}})
        return ToolResult(call=call, content=json.dumps(result, ensure_ascii=False), ok=ok, ms=ms)
