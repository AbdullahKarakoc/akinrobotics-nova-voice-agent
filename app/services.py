"""Builds and holds the long-lived components (models are loaded once per process)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.agent import Agent
from app.audio.vad import SileroVAD
from app.config import Settings
from app.knowledge import create_retriever
from app.llm import create_llm
from app.pipeline.metrics import MetricsStore
from app.stt import SpeechToText, create_stt
from app.tools import create_tool_registry
from app.tts import TextToSpeech, create_tts

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    vad: SileroVAD
    stt: SpeechToText
    tts: TextToSpeech
    agent: Agent
    metrics: MetricsStore
    # One worker each: STT and TTS are CPU-bound and already multi-threaded internally;
    # serialising them avoids oversubscribing the cores the LLM also needs.
    stt_executor: ThreadPoolExecutor
    tts_executor: ThreadPoolExecutor


def build_services(settings: Settings) -> Services:
    timings = {}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = round((time.perf_counter() - start) * 1000)
        return result

    vad = timed("vad", SileroVAD)
    stt = timed("stt", lambda: create_stt(settings))
    tts = timed("tts", lambda: create_tts(settings))
    retriever = timed("retriever", lambda: create_retriever(settings))
    tools = timed("tools", lambda: create_tool_registry(settings))
    agent = Agent(create_llm(settings), retriever, tools, settings)
    log.info("services loaded", extra={"data": {"load_ms": timings, "tools": list(tools.tools)}})
    return Services(
        settings=settings,
        vad=vad,
        stt=stt,
        tts=tts,
        agent=agent,
        metrics=MetricsStore(),
        stt_executor=ThreadPoolExecutor(1, thread_name_prefix="stt"),
        tts_executor=ThreadPoolExecutor(1, thread_name_prefix="tts"),
    )
