"""Helpers for deterministic tool routers (keyword + slot extraction on the user text)."""

from __future__ import annotations

import re
import unicodedata

from app.knowledge.text import turkish_lower

PROVINCES = (
    "Adana Adıyaman Afyonkarahisar Ağrı Aksaray Amasya Ankara Antalya Ardahan Artvin Aydın Balıkesir Bartın Batman "
    "Bayburt Bilecik Bingöl Bitlis Bolu Burdur Bursa Çanakkale Çankırı Çorum Denizli Diyarbakır Düzce Edirne Elazığ "
    "Erzincan Erzurum Eskişehir Gaziantep Giresun Gümüşhane Hakkari Hatay Iğdır Isparta İstanbul İzmir Kahramanmaraş "
    "Karabük Karaman Kars Kastamonu Kayseri Kilis Kırıkkale Kırklareli Kırşehir Kocaeli Konya Kütahya Malatya Manisa "
    "Mardin Mersin Muğla Muş Nevşehir Niğde Ordu Osmaniye Rize Sakarya Samsun Şanlıurfa Siirt Sinop Şırnak Sivas "
    "Tekirdağ Tokat Trabzon Tunceli Uşak Van Yalova Yozgat Zonguldak"
).split()
_LOCATIVE = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,})['’]?(?:da|de|ta|te|daki|deki)\b")
_EN_IN = re.compile(r"\b(?:in|for|at)\s+([A-Z][a-z]{2,})\b")


def fold(text: str) -> str:
    """Case- and accent-insensitive key: 'İstanbul', 'ISTANBUL', 'istanbul' -> 'istanbul'."""
    text = unicodedata.normalize("NFKD", turkish_lower(text).replace("ı", "i"))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip()


def mentions(text: str, pattern: str) -> bool:
    """Regex search on the folded text (write patterns without Turkish diacritics)."""
    return re.search(pattern, fold(text)) is not None


def extract_city(text: str) -> str | None:
    folded = fold(text)
    for city in PROVINCES:
        if re.search(rf"\b{fold(city)}", folded):
            return city
    m = _LOCATIVE.search(text) or _EN_IN.search(text)
    return m.group(1) if m else None
