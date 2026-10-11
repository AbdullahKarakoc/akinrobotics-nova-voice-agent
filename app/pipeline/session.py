"""One WebSocket conversation: mic audio in -> VAD -> STT -> agent -> TTS -> audio out.

Latency design:
* end-of-turn is decided by VAD silence (no fixed push-to-talk delay);
* the LLM answer is streamed and cut into sentences; each sentence is
  synthesised and sent while the LLM keeps generating the next one, so the
  user hears the first sentence long before the answer is complete;
* when the user starts speaking over the agent (barge-in) the running turn is
  cancelled immediately — the HTTP stream to the LLM is closed, which also
  stops generation on the server.

Wire protocol
  client -> server: binary = 16 kHz mono int16 PCM;
                    text   = JSON {"type": "text"|"reset"|"playback", ...}
  server -> client: binary = 8-byte header (uint32 turn_id, uint32 sample_rate) + int16 PCM;
                    text   = JSON events (ready, vad, transcript, sentence, answer, metrics, interrupt, error)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
import uuid
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.agent import AgentTurn, Conversation
from app.agent.prompts import UNVERIFIED_REPLIES
from app.agent.speech_text import SentenceChunker, clean_for_speech, detect_text_language, ungrounded_numbers
from app.audio.vad import Endpointer, EndpointerConfig, VadEventType
from app.logging_setup import session_id_var, turn_id_var
from app.pipeline.metrics import TurnMetrics
from app.services import Services
from app.stt.domain_terms import fix_domain_terms

log = logging.getLogger(__name__)


class VoiceSession:
    def __init__(self, ws: WebSocket, services: Services):
        self.ws = ws
        self.services = services
        self.settings = services.settings
        self.id = uuid.uuid4().hex[:6]
        s = self.settings
        self.endpointer = Endpointer(
            services.vad.stream(),
            EndpointerConfig(
                threshold=s.vad_threshold,
                min_silence_ms=s.vad_min_silence_ms,
                min_speech_ms=s.vad_min_speech_ms,
                speech_pad_ms=s.vad_speech_pad_ms,
                max_utterance_s=s.vad_max_utterance_s,
            ),
        )
        self.conversation = Conversation(s.max_history_turns)
        self.turn_task: asyncio.Task | None = None
        self.turn_counter = 0
        self.client_playing = False
        self._turn_audio: np.ndarray | None = None  # utterance of the running voice turn
        self._turn_spoke = False  # has the running turn sent any audio yet?
        self._carry_audio: np.ndarray | None = None  # see interrupt()
        self._audio_bytes = 0
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------ I/O
    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_audio(self, turn_id: int, sample_rate: int, pcm16: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<II", turn_id, sample_rate) + pcm16)

    async def run(self) -> None:
        session_id_var.set(self.id)
        log.info("session opened")
        await self.send_json(
            {
                "type": "ready",
                "session_id": self.id,
                "sample_rate": 16_000,
                "languages": self.settings.languages,
                "llm": self.services.agent.llm.model,
                "barge_in": self.settings.barge_in,
            }
        )
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await self.on_audio(message["bytes"])
                elif message.get("text"):
                    await self.on_control(json.loads(message["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await self._cancel_turn()
            log.info("session closed")

    # ------------------------------------------------------------ control
    @property
    def turn_running(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    @property
    def agent_busy(self) -> bool:
        return self.turn_running or self.client_playing

    @property
    def agent_speaking(self) -> bool:
        return self.client_playing or (self.turn_running and self._turn_spoke)

    async def on_control(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "text" and str(msg.get("text", "")).strip():
            await self.interrupt("new_text")
            self._start_turn(text=str(msg["text"]).strip())
        elif kind == "playback":
            self.client_playing = bool(msg.get("playing"))
        elif kind == "interrupt":  # the user pressed the mic button while the agent was talking
            await self.interrupt("button")
        elif kind == "mic":  # the user paused / resumed listening
            # A pause can cut an utterance in half; never glue it to what is said after resuming.
            self.endpointer.reset()
            self._carry_audio = None
            log.info(
                "mic " + ("on" if msg.get("on") else "off"),
                extra={"data": {"audio_s_since_last_switch": round(self._audio_bytes / 32000, 1)}},
            )
            self._audio_bytes = 0
        elif kind == "reset":
            await self.interrupt("reset")
            self.conversation.clear()
            self.endpointer.reset()
            await self.send_json({"type": "reset"})

    async def on_audio(self, pcm16: bytes) -> None:
        self._audio_bytes += len(pcm16)  # diagnostics: is the browser actually sending audio?
        speaking = self.agent_speaking
        if speaking and not self.settings.barge_in:
            return  # half-duplex mode: the mic is ignored while the agent talks
        # A stricter onset while our own voice may leak into the mic (echo) avoids self-interruption.
        self.endpointer.min_speech_ms = (
            self.settings.barge_in_min_speech_ms if speaking else self.settings.vad_min_speech_ms
        )
        for event in self.endpointer.process(pcm16):
            if event.type is VadEventType.SPEECH_START:
                if self.agent_busy:
                    await self.interrupt("barge_in")
                await self.send_json({"type": "vad", "state": "speech_start"})
            else:
                await self.send_json({"type": "vad", "state": "speech_end"})
                self._start_turn(audio=event.audio, speech_end_time=event.speech_end_time)

    async def interrupt(self, reason: str) -> None:
        was_busy = self.agent_busy
        if reason == "barge_in" and self.turn_running and not self._turn_spoke and self._turn_audio is not None:
            # The user only paused mid-sentence and went on before we answered:
            # their next utterance continues this one, so keep it and prepend it.
            self._carry_audio = self._turn_audio
        await self._cancel_turn()
        self.client_playing = False
        if was_busy:
            log.info("interrupted", extra={"data": {"reason": reason, "carry": self._carry_audio is not None}})
            await self.send_json({"type": "interrupt", "reason": reason, "turn_id": self.turn_counter})

    async def _cancel_turn(self) -> None:
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.turn_task
        self.turn_task = None

    def _start_turn(
        self, *, audio: np.ndarray | None = None, text: str | None = None, speech_end_time: float | None = None
    ) -> None:
        if audio is not None and self._carry_audio is not None:
            audio = np.concatenate([self._carry_audio, audio])
        self._carry_audio = None
        self._turn_audio = audio
        self._turn_spoke = False
        self.turn_counter += 1
        self.turn_task = asyncio.create_task(
            self._run_turn(self.turn_counter, audio, text, speech_end_time or time.perf_counter())
        )

    # --------------------------------------------------------------- turn
    async def _run_turn(self, turn_id: int, audio: np.ndarray | None, text: str | None, t0: float) -> None:
        turn_id_var.set(str(turn_id))
        loop = asyncio.get_running_loop()
        metrics = TurnMetrics(turn_id=turn_id, mode="voice" if audio is not None else "text")
        settings = self.settings
        agent_turn = AgentTurn()
        question = text or ""
        tts_task: asyncio.Task | None = None
        try:
            if audio is not None:
                stt_start = time.perf_counter()
                metrics.endpoint_ms = (stt_start - t0) * 1000
                transcript = await loop.run_in_executor(
                    self.services.stt_executor, self.services.stt.transcribe, audio, settings.languages
                )
                metrics.audio_s, metrics.stt_ms, metrics.stt_rtf = (
                    transcript.audio_s,
                    transcript.processing_s * 1000,
                    transcript.rtf,
                )
                question, language = fix_domain_terms(transcript.text), transcript.language
                if transcript.language_prob < 0.85:
                    # Short utterances fool Whisper's language ID ("Hello, how are you?" -> tr, p=0.68);
                    # when it is unsure, let the words of the transcript decide.
                    language = detect_text_language(question, default=language)
                log.info(
                    "transcript",
                    extra={"data": {"text": question, "lang": language, "p": transcript.language_prob, "stt_ms": round(metrics.stt_ms)}},
                )
                if not question:
                    await self.send_json({"type": "transcript", "turn_id": turn_id, "text": "", "empty": True})
                    return
            else:
                language = detect_text_language(question, settings.default_language)
            metrics.language = language
            await self.send_json({"type": "transcript", "turn_id": turn_id, "text": question, "language": language})

            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            said: list[str] = []  # what was actually spoken (after the number check)
            tts_task = asyncio.create_task(
                self._speak(sentences, turn_id, language, metrics, t0, agent_turn, said)
            )
            chunker = SentenceChunker()
            async for delta in self.services.agent.respond(question, language, self.conversation, agent_turn):
                for sentence in chunker.push(delta):
                    await sentences.put(sentence)
            for sentence in chunker.flush():
                await sentences.put(sentence)
            await sentences.put(None)
            await tts_task

            await self.send_json(
                {
                    "type": "answer",
                    "turn_id": turn_id,
                    "text": " ".join(said) or agent_turn.spoken.strip(),
                    "language": language,
                    "sources": agent_turn.sources(),
                    "ungrounded_numbers": agent_turn.ungrounded,
                }
            )
        except asyncio.CancelledError:
            metrics.interrupted = True
            if agent_turn.answer.strip():  # keep what was said so a follow-up still makes sense
                self.conversation.add(question, agent_turn.answer.strip() + " …")
            raise
        except Exception as exc:
            log.exception("turn failed")
            await self.send_json({"type": "error", "turn_id": turn_id, "message": str(exc)})
        finally:
            if tts_task and not tts_task.done():
                tts_task.cancel()
            self._finish_metrics(metrics, agent_turn, t0)

    async def _speak(
        self,
        sentences: asyncio.Queue[str | None],
        turn_id: int,
        language: str,
        metrics: TurnMetrics,
        t0: float,
        agent_turn: AgentTurn,
        said: list[str],
    ) -> None:
        loop = asyncio.get_running_loop()
        synth_s = audio_s = 0.0
        withheld = drop_rest = False
        while (sentence := await sentences.get()) is not None:
            if drop_rest:  # the tail of a withheld sentence that was cut at a comma
                drop_rest = sentence.rstrip().endswith(",")
                continue
            # Hallucination guard *before* speech: a sentence with a number that appears in no
            # source (question, retrieved text, tool output) is not spoken.
            if invented := ungrounded_numbers(sentence, agent_turn.evidence_text()):
                log.warning("sentence withheld", extra={"data": {"sentence": sentence, "numbers": invented}})
                drop_rest = sentence.rstrip().endswith(",")
                if withheld:
                    continue
                withheld = True
                sentence = UNVERIFIED_REPLIES.get(language, UNVERIFIED_REPLIES["tr"])
            if sentence.strip() in said:  # small models sometimes loop on a sentence
                log.warning("repeated sentence skipped", extra={"data": {"sentence": sentence}})
                continue
            sentence = sentence.strip()
            said.append(sentence)
            spoken = clean_for_speech(sentence, language)
            if not spoken:
                continue
            await self.send_json({"type": "sentence", "turn_id": turn_id, "text": sentence})
            speech = await loop.run_in_executor(
                self.services.tts_executor, self.services.tts.synthesize, spoken, language
            )
            if metrics.first_audio_ms is None:
                metrics.tts_first_ms = speech.processing_s * 1000
                metrics.first_audio_ms = (time.perf_counter() - t0) * 1000
            synth_s += speech.processing_s
            audio_s += speech.duration_s
            self._turn_spoke = True
            await self.send_audio(turn_id, speech.sample_rate, speech.pcm16)
        if audio_s:
            metrics.tts_rtf = synth_s / audio_s
            metrics.tts_audio_s = audio_s

    def _finish_metrics(self, metrics: TurnMetrics, turn: AgentTurn, t0: float) -> None:
        metrics.total_ms = (time.perf_counter() - t0) * 1000
        metrics.retrieval_ms = turn.retrieval_ms or None
        metrics.llm_ttft_ms = turn.llm_ttft_ms
        metrics.tool_ms = turn.tool_ms or None
        metrics.tools = [r.call.name for r in turn.tool_results]
        if turn.llm_stats:
            metrics.llm_rounds = len(turn.llm_stats)
            metrics.llm_prompt_tokens = sum(s.prompt_tokens for s in turn.llm_stats)
            metrics.llm_completion_tokens = sum(s.completion_tokens for s in turn.llm_stats)
            prefill = [s.prompt_eval_ms for s in turn.llm_stats if s.prompt_eval_ms is not None]
            metrics.llm_prefill_ms = sum(prefill) if prefill else None
            speeds = [s.tokens_per_s for s in turn.llm_stats if s.tokens_per_s]
            metrics.llm_tokens_per_s = sum(speeds) / len(speeds) if speeds else None
        self.services.metrics.add(metrics)
        log.info("turn metrics", extra={"data": metrics.as_dict()})
        if not metrics.interrupted:
            asyncio.get_running_loop().create_task(self._send_quietly({"type": "metrics", **metrics.as_dict()}))

    async def _send_quietly(self, payload: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):  # the socket may already be closed
            await self.send_json(payload)
