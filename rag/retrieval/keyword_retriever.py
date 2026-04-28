from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.config import Settings
from rag.types import DocumentChunk, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-z0-9_/-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=4)
def _load_chunks_from_path(chunks_path: str) -> tuple[DocumentChunk, ...]:
    resolved = Path(chunks_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Chunk file not found at {resolved}. Run ingest first."
        )
    chunks: list[DocumentChunk] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            chunks.append(DocumentChunk(**payload))
    return tuple(chunks)


@lru_cache(maxsize=4)
def _build_bm25_index(chunks_path: str) -> BM25Okapi:
    chunks = _load_chunks_from_path(chunks_path)
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


def clear_keyword_index_cache() -> None:
    _load_chunks_from_path.cache_clear()
    _build_bm25_index.cache_clear()


class KeywordRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks = list(_load_chunks_from_path(str(settings.chunks_path)))
        self.bm25 = _build_bm25_index(str(settings.chunks_path))

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results: list[RetrievedChunk] = []
        for chunk, score in ranked[:top_k]:
            results.append(
                RetrievedChunk(
                    doc=chunk.doc_name,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    category=chunk.category,
                    path=chunk.path,
                    source_type=chunk.source_type,
                    asset_path=chunk.asset_path,
                    asset_kind=chunk.asset_kind,
                )
            )
        return results
