"""Fix the spelling of domain terms in transcripts.

Whisper writes spoken product codes in many ways ("Rehber ar iki", "Rehber ve 2").
Biasing the decoder with a prompt fixes that but hurts general recognition, so
the names are normalised after decoding instead. Add a line per new term.
"""

from __future__ import annotations

import re

_TERMS = [
    # "Rehber R2" as heard in real use: "Rehber ve iki", "Zehbe Z2", "Rehbar R 2", "Vehbar ve iki", "Heddar ve iki"
    (r"\b[rzvh]e(?:h?b|h?dd?)a?e?r?[\s-]*(?:r|ar|er|re|ve|z)?[\s-]*(?:2|iki|two)\b", "Rehber R2"),
    (r"\bpi[ly]\s*[oö]mr[uü]", "pil ömrü"),  # "Piyomru"
    (r"\bta[sş][ıi]y[ıi]c[ıi][\s-]*(?:t|te)?[\s-]*(?:5|beş|bes|five)\b", "Taşıyıcı T5"),
    (r"\btemizlik[\s-]*(?:c|ce|se|si|ci)?[\s-]*(?:3|üç|uc|three)\b", "Temizlik C3"),
    (r"\be[ğg]itim[\s-]*e[\s-]*(?:1|bir|one)\b", "Eğitim E1"),
    (r"\bnova[\s-]*robo(?:tik|tic|tics)\b", "Nova Robotik"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _TERMS]


def fix_domain_terms(text: str) -> str:
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
