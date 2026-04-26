from __future__ import annotations

from collections import defaultdict

from rag.types import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}

    for result_set in result_sets:
        for rank, item in enumerate(result_set, start=1):
            fused_scores[item.chunk_id] += 1.0 / (k + rank)
            by_id[item.chunk_id] = item

    fused = [
        RetrievedChunk(
            doc=item.doc,
            chunk_id=item.chunk_id,
            text=item.text,
            score=fused_scores[item.chunk_id],
            category=item.category,
            path=item.path,
        )
        for item in by_id.values()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)
