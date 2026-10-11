"""FastAPI application: web UI, voice WebSocket, text chat API, metrics and health."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import AgentTurn, Conversation
from app.agent.speech_text import detect_text_language
from app.config import get_settings
from app.logging_setup import setup_logging
from app.pipeline.session import VoiceSession
from app.services import Services, build_services

log = logging.getLogger("app")
settings = get_settings()
setup_logging(settings.log_level, settings.log_json)

state: dict[str, object] = {"llm_ready": False}


async def _warmup(services: Services) -> None:
    """Retry until the LLM server is reachable (it may still be pulling the model)."""
    for attempt in range(60):
        try:
            if hasattr(services.agent.llm, "ensure_model"):
                log.info("checking llm model", extra={"data": {"model": services.agent.llm.model}})
                if await services.agent.llm.ensure_model():
                    log.info("llm model pulled", extra={"data": {"model": services.agent.llm.model}})
            await services.agent.warmup()
            state["llm_ready"] = True
            return
        except Exception as exc:
            log.warning("llm not ready yet", extra={"data": {"attempt": attempt + 1, "error": str(exc)[:200]}})
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = await asyncio.to_thread(build_services, settings)
    app.state.services = services
    warmup = asyncio.create_task(_warmup(services))
    yield
    warmup.cancel()
    await services.agent.llm.aclose()


app = FastAPI(title="Nova Voice Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.middleware("http")
async def no_stale_ui(request: Request, call_next):
    """Make the browser revalidate the UI files on every load (cheap: ETag -> 304), so a new
    app.js is picked up after an update instead of a cached old one."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_ready": state["llm_ready"], "llm_model": settings.llm_model}


@app.get("/api/metrics")
async def metrics() -> dict:
    return app.state.services.metrics.summary()


@app.websocket("/ws")
async def voice(ws: WebSocket) -> None:
    await ws.accept()
    await VoiceSession(ws, app.state.services).run()


class ChatRequest(BaseModel):
    text: str
    language: str | None = None
    session_id: str = "default"


_conversations: collections.OrderedDict[str, Conversation] = collections.OrderedDict()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    """Text-only access to the same agent (used by scripts/evaluate.py and for debugging)."""
    services: Services = app.state.services
    conversation = _conversations.setdefault(req.session_id, Conversation(settings.max_history_turns))
    _conversations.move_to_end(req.session_id)
    while len(_conversations) > 100:
        _conversations.popitem(last=False)

    language = req.language or detect_text_language(req.text, settings.default_language)
    turn = AgentTurn()
    start = time.perf_counter()
    async for _ in services.agent.respond(req.text, language, conversation, turn):
        pass
    return {
        "answer": turn.spoken.strip(),
        "language": language,
        "sources": turn.sources(),
        "ungrounded_numbers": turn.ungrounded,
        "error": turn.error,
        "metrics": {
            "total_ms": round((time.perf_counter() - start) * 1000),
            "retrieval_ms": round(turn.retrieval_ms, 1),
            "llm_ttft_ms": round(turn.llm_ttft_ms) if turn.llm_ttft_ms else None,
            "tool_ms": round(turn.tool_ms),
            "llm_rounds": len(turn.llm_stats),
            "prompt_tokens": sum(s.prompt_tokens for s in turn.llm_stats),
            "completion_tokens": sum(s.completion_tokens for s in turn.llm_stats),
        },
    }
