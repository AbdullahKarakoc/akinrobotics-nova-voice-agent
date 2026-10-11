"""Language-aware text normalisation for lexical search (Turkish + English)."""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[0-9a-zçğıöşüâîû]+")
_STOPWORDS = {
    # tr
    "ve", "ile", "bir", "bu", "şu", "o", "da", "de", "mi", "mı", "mu", "mü", "ne", "nasıl", "nedir",
    "için", "gibi", "kaç", "var", "mı", "ben", "sen", "biz", "siz", "çok", "daha", "en", "ya", "veya",
    "ki", "ise", "olan", "olarak", "hangi", "neler", "midir", "mıdır", "acaba", "lütfen",
    # en
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be", "do",
    "does", "what", "how", "which", "can", "i", "you", "we", "it", "this", "that", "with", "my", "your",
    "please", "about", "there",
}  # fmt: skip


def turkish_lower(text: str) -> str:
    """Lower-case with Turkish dotted/dotless I handled correctly (str.lower() gets 'I' -> 'i' wrong)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str, stem_len: int = 5) -> list[str]:
    """Tokens for BM25. Truncating to the first 5 characters is a simple, well-known and effective
    stemmer for agglutinative Turkish ("robotun", "robotlar" -> "robot") and harmless for English."""
    text = unicodedata.normalize("NFC", turkish_lower(text))
    return [t[:stem_len] for t in _TOKEN.findall(text) if t not in _STOPWORDS]
