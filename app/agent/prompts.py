"""Prompt templates.

The system prompt is static (no per-turn data) so the LLM server can keep its
KV cache: together with the tool schemas it is the largest part of the
prompt, and re-processing it every turn would cost seconds on CPU. Per-turn
data (retrieved context, pre-fetched tool data, reply language) goes into the
last user message.

Two knowledge modes:
* "rag"  - the top retrieved chunks are put into each user message;
* "full" - the whole (small) knowledge base is placed in the system prompt once
           and stays in the KV cache, so each turn only processes the question
           (cache-augmented generation). Much lower latency on CPU, only viable
           while the corpus fits comfortably in the context window.
"""

from __future__ import annotations

from app.knowledge import Chunk, Hit
from app.tools import ToolResult

SYSTEM_PROMPT = """You are Nova, the voice assistant of Nova Robotik, a robotics company. Your words are converted to speech.

Rules:
1. Facts about the company, its products, policies, orders, prices, stock and live data (weather, time) must come ONLY from the KNOWLEDGE / CONTEXT text, LIVE DATA or tool results you are given. Never use your own memory for these.
2. If they do not contain the answer, say you don't have that information and mention what you can help with (Nova Robotik products, warranty, service, orders, prices, weather). Never guess. Never invent names, numbers or dates. Never take a number that belongs to a different product or topic.
3. Use tools for order status, product price or stock, service center addresses, weather and the current date or time, unless LIVE DATA already has the answer.
4. Do not answer questions unrelated to Nova Robotik (capitals, history, other companies, sports, celebrities, coding, news), even if you know the answer: politely say you can only help with Nova Robotik topics. Greetings and thanks are fine.
5. Speak naturally and briefly: one to three short sentences. No markdown, no lists, no emojis, no links. Do not mention "context", "knowledge base", "documents" or "tools".
6. Reply in the language the user message asks for.
7. The user's words come from speech recognition and may contain mistakes. If a word looks like a misheard product or term (for example "Zehbe Z2" or "Rfb2" for Rehber R2), treat it as the closest one in the CONTEXT and use the correct name in your answer."""

# Spoken instead of a sentence that contains a number found in no source (see session._speak).
UNVERIFIED_REPLIES = {
    "tr": "Bu konuda kaynaklarımda kesin bir bilgi yok.",
    "en": "I don't have verified information on that.",
}

# Written in the target language: small models follow (and get primed by) it more reliably.
LANGUAGE_INSTRUCTIONS = {"tr": "(Yalnızca Türkçe cevap ver.)", "en": "(Answer only in English.)"}

TOOL_FILLERS = {"tr": "Bir saniye, kontrol ediyorum.", "en": "One moment, let me check."}

ERROR_REPLIES = {
    "tr": "Üzgünüm, şu anda bir sorun yaşıyorum. Lütfen biraz sonra tekrar deneyin.",
    "en": "Sorry, I'm having a problem right now. Please try again in a moment.",
}


def _passage(label: str, chunk: Chunk) -> str:
    # Source title after the text: small models tend to read out whatever comes first.
    return f"[{label}] {chunk.text} (source: {chunk.title})"


def build_system_prompt(knowledge: list[Chunk] | None = None) -> str:
    if not knowledge:
        return SYSTEM_PROMPT
    kb = "\n".join(_passage(f"K{i}", c) for i, c in enumerate(knowledge, 1))
    return f"{SYSTEM_PROMPT}\n\nKNOWLEDGE:\n{kb}"


def build_user_message(
    question: str,
    language: str,
    hits: list[Hit] | None,
    tool_results: list[ToolResult] | None = None,
    relevant: list[str] | None = None,
) -> str:
    parts = []
    if hits is not None:  # rag mode: the passages themselves
        context = "\n".join(_passage(str(i), h.chunk) for i, h in enumerate(hits, 1))
        parts.append(f"CONTEXT:\n{context or '(no relevant information found)'}")
    if relevant is not None:  # full mode: only point at the cached passages (a few tokens)
        found = ", ".join(relevant) if relevant else "none - the KNOWLEDGE does not cover this question"
        parts.append(f"RELEVANT KNOWLEDGE: {found}")
    if tool_results:
        data = "\n".join(f"- {r.call.name}({r.call.arguments}): {r.content}" for r in tool_results)
        parts.append(f"LIVE DATA (already fetched for this question):\n{data}")
    parts.append(f"USER: {question}")
    parts.append(LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["tr"]))
    return "\n\n".join(parts)
