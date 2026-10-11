"""Tools served by external MCP (Model Context Protocol) servers, registered like local ones.

Each configured server gets one long-lived MCP session on its own background event loop, so
it can be opened from the synchronous start-up code and outlives any single request. Its
tools are discovered with `tools/list` and appear in the registry next to the local ones:
the LLM sees the same JSON schemas and calls go through the same timeout / error handling.

Routers stay on the agent side (keyed by tool name): they are part of the agent's latency
strategy, not of the data source. Per server, MCPServerConfig picks which tools to take,
renames them, hides them from the LLM (router-only) and trims long results. A server that
cannot be started is logged and skipped, so the agent still runs with its local tools.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import shlex
import sys
import threading
from typing import Any

from app.config import ROOT_DIR, MCPServerConfig
from app.tools import wikipedia
from app.tools.registry import Router, Tool, ToolRegistry
from app.tools.routing import mentions

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ routers
# Serial numbers have 5 digits (R2-24011), service tickets and orders 4 (5002, 1003), so the
# number length keeps these routers apart from each other and from the order router.
_WARRANTY = r"garanti|warrant|bakim|maintenance"
_DIAGNOSTICS = (
    r"calismiyor|calismi|hata|ariza|sarj|batarya|\bpil\b|cevrim ?(dis|ic)|baglan|teshis|ne durumda|"
    r"not working|error|fault|battery|offline|online|diagnos"
)


def spoken_number(text: str, min_digits: int, max_digits: int) -> str | None:
    """First number of the given length; STT may split it ('24 011', '24.011')."""
    for m in re.finditer(r"\d[\d.\s]*\d|\d", text):
        digits = re.sub(r"\D", "", m.group())
        if min_digits <= len(digits) <= max_digits:
            return digits
    return None


def route_warranty(text: str) -> dict | None:
    serial = spoken_number(text, 5, 7)  # 'R2 24011' -> '224011': the server matches the last 5 digits
    return {"serial_number": serial[-5:]} if serial and mentions(text, _WARRANTY) else None


def route_diagnostics(text: str) -> dict | None:
    serial = spoken_number(text, 5, 7)
    if not serial:
        return None
    # A plain "status" question is a diagnostics one unless it is about the warranty.
    if mentions(text, _DIAGNOSTICS) or (mentions(text, r"durum|status") and not mentions(text, _WARRANTY)):
        return {"serial_number": serial[-5:]}
    return None


def route_ticket(text: str) -> dict | None:
    if not mentions(text, r"tale[pb]|kayd|kayit|basvuru|ticket|request|case"):  # talep -> talebim
        return None
    number = spoken_number(text, 4, 4)
    return {"ticket_number": number} if number else None


ROUTERS: dict[str, Router] = {
    "check_warranty": route_warranty,
    "get_robot_diagnostics": route_diagnostics,
    "get_service_ticket_status": route_ticket,
    **wikipedia.ROUTERS,
}


# ------------------------------------------------------------- connection
def _server_target(spec: str) -> Any:
    """'http(s)://...' -> Streamable HTTP; anything else is a command line run over stdio
    ('python' means this interpreter, so the server shares the venv / image)."""
    if spec.startswith(("http://", "https://")):
        return spec
    from mcp import StdioServerParameters

    command, *args = shlex.split(spec, posix=os.name != "nt")
    if command == "python":
        command = sys.executable
    return StdioServerParameters(command=command, args=args, cwd=ROOT_DIR)


def _result_payload(result: Any) -> dict[str, Any]:
    text = " ".join(c.text for c in result.content if getattr(c, "type", None) == "text")
    if result.is_error:
        return {"error": f"{text} Tell the user the information is unavailable right now."}
    if isinstance(result.structured_content, dict):
        return result.structured_content
    try:
        payload = json.loads(text)
    except ValueError:
        payload = text
    return payload if isinstance(payload, dict) else {"result": payload}


def trim_text(value: Any, max_chars: int) -> Any:
    """Shortens long strings (recursively) at the last sentence end within `max_chars`."""
    if isinstance(value, dict):
        return {k: trim_text(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [trim_text(v, max_chars) for v in value]
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    cut = value[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > max_chars // 3 else cut.rsplit(" ", 1)[0] + " ..."


class MCPConnection:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.spec = config.target
        self.tools: list[Any] = []
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self, timeout_s: float) -> None:
        """Connect and list the tools; raises if the server is not usable within the timeout."""
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()

        def run() -> None:
            # Subprocesses on Windows need the proactor loop, whatever policy the host app set.
            loop = asyncio.ProactorEventLoop() if os.name == "nt" else asyncio.new_event_loop()  # type: ignore[attr-defined]
            self._loop = loop
            try:
                loop.run_until_complete(self._serve(ready))
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name="mcp-client", daemon=True)
        self._thread.start()
        try:
            ready.result(timeout_s)
        except BaseException:
            self.close()
            raise

    async def _serve(self, ready: concurrent.futures.Future[None]) -> None:
        from mcp import Client

        self._stop = asyncio.Event()
        try:
            async with Client(_server_target(self.spec)) as client:
                self.tools = list((await client.list_tools()).tools)
                self._client = client
                ready.set_result(None)
                await self._stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                log.warning("mcp server disconnected", extra={"data": {"server": self.spec, "error": repr(exc)}})
        finally:
            self._client = None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None or self._loop is None:
            raise ConnectionError(f"MCP server '{self.spec}' is not connected")
        future = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, arguments), self._loop)
        payload = _result_payload(await asyncio.wrap_future(future))
        return trim_text(payload, self.config.max_chars) if self.config.max_chars else payload

    def close(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:  # loop closed in between
                pass
        if self._thread:
            self._thread.join(timeout=5)


def _handler(conn: MCPConnection, name: str):
    async def call(**arguments: Any) -> dict[str, Any]:
        return await conn.call(name, arguments)

    return call


def _start(conn: MCPConnection, timeout_s: float) -> bool:
    try:
        conn.start(timeout_s)
        return True
    except Exception as exc:
        log.warning("mcp server unavailable, its tools are skipped", extra={"data": {"server": conn.spec, "error": repr(exc)}})
        return False


def register(registry: ToolRegistry, servers: list[str | MCPServerConfig], timeout_s: float = 30.0) -> None:
    conns = [MCPConnection(s if isinstance(s, MCPServerConfig) else MCPServerConfig(target=s)) for s in servers]
    if not conns:
        return
    with concurrent.futures.ThreadPoolExecutor(len(conns)) as pool:  # servers start in parallel
        started = list(pool.map(lambda c: _start(c, timeout_s), conns))
    for conn, ok in zip(conns, started):
        if not ok:
            continue
        registry.on_close(conn.close)
        cfg, added = conn.config, []
        for t in conn.tools:
            name = cfg.prefix + t.name
            if cfg.tools is not None and t.name not in cfg.tools:
                continue
            if name in registry.tools:
                log.warning("mcp tool name already taken, skipped", extra={"data": {"server": conn.spec, "tool": name}})
                continue
            schema = t.input_schema or {}
            # Pydantic adds a "title" per property: noise for the LLM and extra prompt tokens.
            props = {k: {f: v for f, v in p.items() if f != "title"} for k, p in schema.get("properties", {}).items()}
            required = list(schema.get("required", []))
            handler = _handler(conn, t.name)
            registry.add(Tool(name, t.description or "", props, required, handler, ROUTERS.get(name), cfg.llm_visible))
            added.append(name)
        log.info("mcp tools registered", extra={"data": {"server": conn.spec, "tools": added}})
