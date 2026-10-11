"""End-to-end voice latency benchmark through the real WebSocket pipeline.

    python -m scripts.benchmark_voice [--url ws://localhost:8000/ws] [--out reports/voice.json]

Spoken questions are synthesised with Piper, then streamed to the server in
real time (32 ms packets, like the browser does), followed by silence. The
client measures the time from the last speech sample it sent to the first
agent audio packet it receives: that is the latency the user experiences
(VAD end-of-turn wait + STT + retrieval/tools + LLM first sentence + TTS).
Server-side breakdowns (STT/TTS RTF, LLM TTFT, ...) come from the `metrics` event.

Note: synthetic TTS speech is cleaner than a real microphone, so STT accuracy
here is optimistic for clarity but pessimistic for the robotic Turkish voice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.tts import create_tts  # noqa: E402

QUESTIONS = [
    ("tr", "Rehber R2 robotunun pil ömrü ne kadar?"),
    ("tr", "Garanti süresi kaç yıl?"),
    ("tr", "Ankara'da hava nasıl?"),
    ("tr", "Bin üç numaralı siparişim ne durumda?"),
    ("tr", "Robot kiralayabilir miyim?"),
    ("en", "What is the warranty period for the cleaning robot?"),
    ("en", "How often should maintenance be done?"),
    ("en", "Can I rent a robot?"),
]
PACKET = 512  # samples at 16 kHz = 32 ms


def to_16k(pcm16: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)


async def run_turn(ws, audio: np.ndarray, timeout_s: float = 120) -> dict:
    lead = np.zeros(int(0.3 * 16000), dtype=np.int16)
    speech = np.concatenate([lead, audio])
    result: dict = {}
    first_audio = asyncio.get_running_loop().create_future()
    done = asyncio.get_running_loop().create_future()

    async def reader():
        async for message in ws:
            if isinstance(message, bytes):
                if not first_audio.done():
                    first_audio.set_result(time.perf_counter())
                turn_id, rate = struct.unpack("<II", message[:8])
                result["audio_s"] = result.get("audio_s", 0) + (len(message) - 8) / 2 / rate
                continue
            event = json.loads(message)
            if event["type"] == "transcript":
                result["transcript"] = event.get("text", "")
                result["language"] = event.get("language")
                if event.get("empty") and not done.done():
                    done.set_result(None)
            elif event["type"] == "answer":
                result["answer"] = event["text"]
            elif event["type"] == "metrics":
                result["server"] = event
                if not done.done():
                    done.set_result(None)

    reader_task = asyncio.create_task(reader())
    start = time.perf_counter()
    for i in range(0, len(speech), PACKET):  # real-time pacing
        await ws.send(speech[i : i + PACKET].tobytes())
        target = start + (i + PACKET) / 16000
        await asyncio.sleep(max(0, target - time.perf_counter()))
    speech_end = time.perf_counter()
    silence = np.zeros(PACKET, dtype=np.int16).tobytes()
    deadline = speech_end + timeout_s
    while not done.done() and time.perf_counter() < deadline:
        await ws.send(silence)  # keep streaming silence like an open microphone
        await asyncio.sleep(PACKET / 16000)
    if first_audio.done():
        result["voice_to_voice_ms"] = (first_audio.result() - speech_end) * 1000
    await asyncio.sleep(0.5)
    reader_task.cancel()
    return result


def summarize(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p90": round(values[min(len(values) - 1, int(0.9 * len(values)))], 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--out", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    tts = create_tts(get_settings())
    clips = []
    for lang, text in QUESTIONS:
        speech = tts.synthesize(text, lang)
        clips.append((lang, text, to_16k(speech.pcm16, speech.sample_rate)))

    rows = []
    async with websockets.connect(args.url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print(f"connected: llm={ready['llm']}\n")
        for _ in range(args.repeat):
            for lang, text, audio in clips:
                r = await run_turn(ws, audio)
                r.update({"question": text, "expected_language": lang})
                rows.append(r)
                s = r.get("server", {})
                print(
                    f"{r.get('voice_to_voice_ms', float('nan')):7.0f} ms | stt {s.get('stt_ms', 0):5.0f} "
                    f"| llm_ttft {s.get('llm_ttft_ms', 0):6.0f} | tts1 {s.get('tts_first_ms', 0):4.0f} "
                    f"| [{r.get('language')}] {r.get('transcript')!r}\n{'':10}-> {r.get('answer')}"
                )
                await ws.send(json.dumps({"type": "reset"}))
                await asyncio.sleep(1.0)

    server = [r["server"] for r in rows if "server" in r]
    summary = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "llm": ready["llm"],
        "turns": len(rows),
        "voice_to_voice_ms": summarize([r["voice_to_voice_ms"] for r in rows if "voice_to_voice_ms" in r]),
        "language_correct": f"{sum(r.get('language') == r['expected_language'] for r in rows)}/{len(rows)}",
    }
    for key in ("endpoint_ms", "stt_ms", "stt_rtf", "retrieval_ms", "tool_ms", "llm_ttft_ms", "llm_tokens_per_s", "tts_first_ms", "tts_rtf", "first_audio_ms"):
        summary[key] = summarize([s[key] for s in server if s.get(key) is not None])
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "turns": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
