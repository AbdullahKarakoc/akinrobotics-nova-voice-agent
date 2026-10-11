"""Load knowledge-base documents and split them into retrievable chunks.

Any `.md` or `.txt` file dropped into `data/docs/` becomes a knowledge source.
Markdown headings are kept as a breadcrumb prefix on each chunk so a chunk is
self-explanatory both for the embedder and for the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file>#<n>", used as a citation
    source: str  # file name
    title: str  # breadcrumb, e.g. "Ürün Kataloğu > Rehber R2"
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n|(?<=[.!?])\s+", text):
        if current and len(current) + len(para) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        current += para + " "
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(text: str, source: str, max_chars: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[str] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        body.clear()
        if not content:
            return
        title = " > ".join(h for h in headings if h)
        for piece in _split_long(content, max_chars):
            chunks.append(Chunk(id=f"{source}#{len(chunks) + 1}", source=source, title=title, text=piece))

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            headings[:] = headings[: level - 1] + [""] * max(0, level - 1 - len(headings))
            headings.append(m.group(2).strip())
        else:
            body.append(line)
    flush()
    return chunks


def load_chunks(docs_dir: Path, max_chars: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, max_chars))
    return chunks
