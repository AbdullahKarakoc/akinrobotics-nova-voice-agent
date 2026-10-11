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
from app.logging_setup import banner, setup_logging
from app.pipeline.session import VoiceSession
from app.services import Services, build_services

log = logging.getLogger("app")
settings = get_settings()
setup_logging(settings.log_level, settings.log_json)


async def _warmup(services: Services) -> None:
    """Retry until the LLM server is reachable (it may still be pulling the model)."""
    agent, llm = services.agent, services.agent.llm
    for attempt in range(60):
        try:
            if hasattr(llm, "has_model") and not await llm.has_model():
                agent.llm_status = "downloading"
                banner(
                    log,
                    f"LLM İNDİRİLİYOR: {llm.model}",
                    "İlk açılışta bir kez (~2 GB). Bitene kadar ajan soru cevaplayamaz;",
                    "ilerleme aşağıda %10'luk adımlarla yazılır.",
                )
                await llm.pull_model()
                log.info("llm model pulled", extra={"data": {"model": llm.model}})
            agent.llm_status = "loading"
            log.info("loading llm into memory", extra={"data": {"model": llm.model}})
            await agent.warmup()
            agent.llm_status = "ready"
            banner(log, "NOVA HAZIR  ->  http://localhost:8000", f"LLM: {llm.model}")
            return
        except Exception as exc:
            banner(
                log,
                f"LLM HAZIR DEĞİL (deneme {attempt + 1}/60, 10 sn sonra tekrar)",
                f"Hata: {str(exc)[:150]}",
                "Ollama açık mı? Model adı doğru mu? İnternet / disk alanı yeterli mi?",
                level=logging.WARNING,
            )
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = await asyncio.to_thread(build_services, settings)
    app.state.services = services
    warmup = asyncio.create_task(_warmup(services))
    yield
    warmup.cancel()
    await services.agent.llm.aclose()
    await asyncio.to_thread(services.agent.tools.close)


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
    status = app.state.services.agent.llm_status
    return {"status": "ok", "llm_ready": status == "ready", "llm_status": status, "llm_model": settings.llm_model}


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
