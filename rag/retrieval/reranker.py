from __future__ import annotations

from rag.types import RetrievedChunk


def maybe_rerank(results: list[RetrievedChunk], enabled: bool = False) -> list[RetrievedChunk]:
    return results if enabled else results
