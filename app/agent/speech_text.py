"""Text utilities between the LLM and TTS: streaming sentence segmentation,
speech-friendly normalisation, text language ID and a lexical grounding check."""

from __future__ import annotations

import re

from app.knowledge.text import turkish_lower

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_SOFT_BREAK = re.compile(r"[,;:]\s+")
_ABBREVIATIONS = ("dr.", "prof.", "vb.", "örn.", "bkz.", "no.", "mr.", "mrs.", "e.g.", "i.e.", "vs.", "st.")


class SentenceChunker:
    """Cuts a token stream into speakable pieces as early as possible.

    The first audio can start as soon as the first sentence is complete instead
    of waiting for the whole answer. The very first piece may even end at a
    comma once it is `first_min_chars` long: that is the piece the user is
    waiting for, and a shorter one means fewer LLM tokens and a faster TTS call
    before the agent starts talking. Later run-on sentences are cut at a comma
    only when they exceed `soft_max_chars`.
    """

    def __init__(self, soft_max_chars: int = 140, first_min_chars: int = 30):
        self.soft_max_chars = soft_max_chars
        self.first_min_chars = first_min_chars
        self.buffer = ""
        self.emitted = False

    def push(self, delta: str) -> list[str]:
        self.buffer += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            piece, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:]
            if piece:
                out.append(piece)
                self.emitted = True
        return out

    def flush(self) -> list[str]:
        piece, self.buffer = self.buffer.strip(), ""
        return [piece] if piece else []

    def _find_cut(self) -> int | None:
        for m in _SENTENCE_END.finditer(self.buffer):
            before = self.buffer[: m.start()].lower()
            if before.endswith(_ABBREVIATIONS):
                continue
            return m.end()
        if not self.emitted:
            for m in _SOFT_BREAK.finditer(self.buffer):
                if m.start() >= self.first_min_chars:
                    return m.end()
        if len(self.buffer) > self.soft_max_chars:
            soft = list(_SOFT_BREAK.finditer(self.buffer))
            if soft:
                return soft[-1].end()
        return None


_MARKDOWN = re.compile(r"[*_`#>]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️]")
_CITATION = re.compile(r"\[\d+\]")

_UNITS = {
    "tr": [(r"(\d)\s*°\s*C\b", r"\1 derece"), (r"%\s*(\d+)", r"yüzde \1"), (r"(\d+)\s*%", r"yüzde \1"),
           (r"\bm²|\bm2\b", "metrekare"), (r"(\d)\s*(?:TL|₺)", r"\1 lira"), (r"\bkm/sa\b|\bkm/h\b", "kilometre"),
           (r"@", " et ")],
    "en": [(r"(\d)\s*°\s*C\b", r"\1 degrees"), (r"(\d+)\s*%", r"\1 percent"), (r"\bm²|\bm2\b", "square meters"),
           (r"(\d)\s*(?:TL|₺)", r"\1 lira"), (r"\bkm/h\b", "kilometers per hour"), (r"@", " at ")],
}  # fmt: skip


def clean_for_speech(text: str, language: str) -> str:
    text = _URL.sub("", text)
    text = _CITATION.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    for pattern, repl in _UNITS.get(language, []):
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()


_TR_CHARS = set("çğışöüÇĞİŞÖÜ")
_TR_WORDS = {
    "ve", "bir", "bu", "ne", "nasıl", "mi", "mı", "mu", "için", "var", "yok", "merhaba", "evet", "hayır", "kaç",
    "nedir", "ben", "sen", "selam", "teşekkürler", "ile", "da", "de", "çok", "daha", "olarak", "olan", "kadar",
    "saat", "yıl", "şu", "anda", "değil", "bilgi", "lütfen", "sizin", "size", "robotu", "hangi",
}  # fmt: skip
_EN_WORDS = {
    "the", "and", "is", "are", "what", "how", "you", "i", "my", "hello", "hi", "thanks", "yes", "no", "do", "does",
    "can", "it", "of", "to", "in", "for", "with", "your", "have", "has", "be", "will", "this", "that", "on", "at",
    "hours", "years", "please", "sorry", "about",
}  # fmt: skip


def detect_text_language(text: str, default: str = "tr") -> str:
    """Cheap TR/EN identification for typed input (spoken input gets it from Whisper).

    Function words decide; Turkish letters only break ties, because English
    answers routinely contain Turkish product names ("Taşıyıcı T5")."""
    # Drop apostrophe suffixes first: "T5'in" must not count as the English word "in".
    words = re.findall(r"\w+", re.sub(r"['’]\w+", "", turkish_lower(text)))
    tr = sum(w in _TR_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if tr != en:
        return "tr" if tr > en else "en"
    if any(ch in _TR_CHARS for ch in text):
        return "tr"
    return default


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[.,]", "", n) for n in _NUMBER.findall(text)}


def ungrounded_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the evidence (context, tool output, question).

    A cheap post-hoc hallucination signal: in this domain most harmful
    hallucinations are invented figures (prices, dates, specs)."""
    known = _numbers(evidence)
    return sorted(n for n in _numbers(answer) if n not in known and len(n) > 1)
