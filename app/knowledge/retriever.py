"""Hybrid (dense + BM25) retriever over the document chunks.

* Dense: multilingual sentence embeddings (fastembed / ONNX, no PyTorch), so a
  Turkish question can match an English document and vice versa.
* Lexical: BM25 with Turkish-aware tokenisation; catches exact names and codes
  ("T5", "E02") that embeddings blur.
* A relevance threshold turns "nothing relevant" into an explicit empty result,
  which the agent uses to refuse instead of guessing.

The corpus is small, so a NumPy matrix is the vector store; swapping in
Qdrant/pgvector only requires reimplementing `search`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.knowledge.documents import Chunk
from app.knowledge.text import tokenize

log = logging.getLogger(__name__)

LEXICAL_BONUS = 0.15  # max score added for a strong BM25 match


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float
    bm25: float


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.len.mean()) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg_len)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        # E5-family models expect role prefixes.
        self._e5 = "e5" in model_name.lower()

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"passage: {t}" if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"query: {text}" if self._e5 else text])[0]


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25([tokenize(c.full_text) for c in chunks])
        self.matrix = self._load_or_embed(cache_dir)
        log.info("retriever ready", extra={"data": {"chunks": len(chunks), "model": embedder.model_name}})

    def _load_or_embed(self, cache_dir: Path | None) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 1), dtype=np.float32)
        digest = hashlib.sha1(self.embedder.model_name.encode())
        for c in self.chunks:
            digest.update(c.full_text.encode())
        cache_file = cache_dir / f"index-{digest.hexdigest()[:16]}.npy" if cache_dir else None
        if cache_file and cache_file.exists():
            return np.load(cache_file)
        start = time.perf_counter()
        matrix = self.embedder.embed_documents([c.full_text for c in self.chunks])
        log.info("embedded corpus", extra={"data": {"chunks": len(self.chunks), "ms": round((time.perf_counter() - start) * 1000)}})
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, matrix)
        return matrix

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        dense = self.matrix @ self.embedder.embed_query(query)
        bm25 = self.bm25.scores(tokenize(query))
        lexical = bm25 / (bm25 + 4.0)  # saturate to [0, 1) without per-query max-normalisation
        scores = dense + LEXICAL_BONUS * lexical
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), dense=float(dense[i]), bm25=float(bm25[i]))
            for i in order
            if scores[i] >= min_score
        ]
