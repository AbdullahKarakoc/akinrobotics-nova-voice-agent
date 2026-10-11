from __future__ import annotations

from app.config import Settings
from app.knowledge.documents import Chunk, load_chunks
from app.knowledge.retriever import Embedder, Hit, Retriever


def create_retriever(settings: Settings) -> Retriever:
    chunks = load_chunks(settings.docs_dir, settings.rag_chunk_chars)
    embedder = Embedder(settings.embedding_model, settings.models_dir / "embeddings")
    return Retriever(chunks, embedder, cache_dir=settings.models_dir / "index_cache")


__all__ = ["Chunk", "Embedder", "Hit", "Retriever", "create_retriever", "load_chunks"]
