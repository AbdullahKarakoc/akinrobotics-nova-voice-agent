"""Routers for the public Wikipedia MCP server (wikipedia-mcp, one instance per language).

Wikipedia is used for one job only: explaining robotics / technology terms a customer meets
in our answers and documents ("LiDAR nedir?", "What does SLAM mean?"). The terms and their
exact article titles are a fixed glossary because
* the server's `get_summary` needs the exact title ("LiDAR" is not found, "LIDAR" is), and its
  search returns off-topic pages ("lidar" -> "Lidar Höyük", an archaeological mound);
* the LLM never sees these tools, so it cannot turn the assistant into a general-knowledge bot
  and the off-topic refusals stay as they were; web text only enters for known articles.
"""

from __future__ import annotations

import re

from app.tools.routing import fold, mentions

# folded aliases (regex) -> (Turkish title or None, English title). Titles checked on 2026-10-11;
# a term without a Turkish article is answered from the English one.
GLOSSARY: dict[str, tuple[str | None, str]] = {
    r"lidar": ("LIDAR", "Lidar"),
    r"slam": (None, "Simultaneous localization and mapping"),
    r"insansi robot|humanoid( robot)?": ("İnsansı robot", "Humanoid robot"),
    r"otonom (mobil )?robot|autonomous (mobile )?robot|\bamr": (None, "Autonomous robot"),
    r"ultrasonik sensor|ultrasonic sensor": (None, "Ultrasonic sensor"),
    r"sensor|algilayici": ("Sensör", "Sensor"),
    r"yapay zeka|artificial intelligence": ("Yapay zekâ", "Artificial intelligence"),
    r"makine ogren(imi|mesi)|machine learning": ("Makine öğrenimi", "Machine learning"),
    r"bilgisayarli goru|computer vision": ("Bilgisayarlı görü", "Computer vision"),
    r"dogal dil isleme|natural language processing|\bnlp": ("Doğal dil işleme", "Natural language processing"),
    r"konusma tanima|speech recognition": ("Konuşma tanıma", "Speech recognition"),
    r"lityum[ -]?iyon|lithium[ -]?ion|li-?ion": ("Lityum iyon pil", "Lithium-ion battery"),
    r"wi-?fi": ("Wi-Fi", "Wi-Fi"),
    r"bluetooth": ("Bluetooth", "Bluetooth"),
    r"robotik|robotics": ("Robotik", "Robotics"),
    r"servo( ?motor)?": ("Servo motor", "Servomotor"),
    r"\bota\b|kablosuz guncelleme|over[ -]the[ -]air": (None, "Over-the-air update"),
}

_ASK_TR = r"\b(nedir|ne demek|ne ise yarar|nasil calisir|hakkinda bilgi|anlatir misin|acikla)"
_ASK_EN = r"\b(what (is|are|does)|what's|how does|tell me about|explain|meaning of)"
# "Rehber R2'nin sensörleri nedir?" is about our product: documents answer that, not Wikipedia.
_OUR_PRODUCTS = r"rehber|tasiyici|temizlik|egitim kit|\b(r2|t5|c3|e1)\b|nova|robotunuz|robotum|sizin"


def _term(text: str) -> tuple[str | None, str] | None:
    folded = fold(text)
    for alias, titles in GLOSSARY.items():  # specific entries come first ("ultrasonik sensor")
        if re.search(rf"\b(?:{alias})", folded):
            return titles
    return None


def _wanted(text: str) -> tuple[str, str] | None:
    """(language, title) when the user asks what a glossary term is."""
    if mentions(text, _OUR_PRODUCTS):
        return None
    ask_tr, ask_en = mentions(text, _ASK_TR), mentions(text, _ASK_EN)
    if not (ask_tr or ask_en) or (titles := _term(text)) is None:
        return None
    tr, en = titles
    return ("tr", tr) if ask_tr and tr else ("en", en)


def route_tr(text: str) -> dict | None:
    wanted = _wanted(text)
    return {"title": wanted[1]} if wanted and wanted[0] == "tr" else None


def route_en(text: str) -> dict | None:
    wanted = _wanted(text)
    return {"title": wanted[1]} if wanted and wanted[0] == "en" else None


ROUTERS = {"wikipedia_tr_get_summary": route_tr, "wikipedia_en_get_summary": route_en}
