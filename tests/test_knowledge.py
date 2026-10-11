import numpy as np

from app.knowledge.documents import chunk_markdown
from app.knowledge.retriever import BM25, Retriever
from app.knowledge.text import tokenize, turkish_lower

DOC = """# Ürün Kataloğu

## Rehber R2
Karşılama robotu.

### Teknik Özellikler
Pil ömrü 8 saattir.

## Taşıyıcı T5
Lojistik robotu, 150 kg taşır.
"""


def test_turkish_lower_handles_dotted_and_dotless_i():
    assert turkish_lower("IŞIK İSTANBUL") == "ışık istanbul"


def test_tokenize_stems_turkish_suffixes():
    assert tokenize("Robotların pilleri") == ["robot", "pille"]
    assert tokenize("Robotun") == tokenize("robotlar") == ["robot"]
    assert "ve" not in tokenize("pil ve şarj")


def test_chunk_markdown_keeps_heading_breadcrumbs():
    chunks = chunk_markdown(DOC, "katalog.md")
    titles = [c.title for c in chunks]
    assert titles == [
        "Ürün Kataloğu > Rehber R2",
        "Ürün Kataloğu > Rehber R2 > Teknik Özellikler",
        "Ürün Kataloğu > Taşıyıcı T5",
    ]
    assert chunks[1].id == "katalog.md#2"
    assert chunks[1].full_text.endswith("Pil ömrü 8 saattir.")


def test_bm25_prefers_documents_with_query_terms():
    docs = [tokenize(t) for t in ["pil ömrü 8 saat", "taşıma kapasitesi 150 kg", "garanti iki yıl"]]
    scores = BM25(docs).scores(tokenize("pil ömrü"))
    assert int(np.argmax(scores)) == 0
    assert scores[1] == scores[2] == 0


class FakeEmbedder:
    """Bag-of-letters embedding: deterministic and good enough to test ranking/thresholds."""

    model_name = "fake"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for ch in turkish_lower(text):
            v[ord(ch) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def test_retriever_threshold_returns_nothing_for_unrelated_queries():
    retriever = Retriever(chunk_markdown(DOC, "katalog.md"), FakeEmbedder())
    hits = retriever.search("Rehber R2 pil ömrü", top_k=2)
    assert hits[0].chunk.title.endswith("Teknik Özellikler")
    assert retriever.search("Rehber R2 pil ömrü", top_k=2, min_score=10.0) == []
