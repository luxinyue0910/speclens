from __future__ import annotations

from rag.config import Settings
from rag.retrieval.keyword_retriever import KeywordRetriever
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.retrieval.vector_retriever import VectorRetriever
from rag.types import RetrievedChunk


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.keyword = KeywordRetriever(settings)
        self.vector = VectorRetriever(settings)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        keyword_results = self.keyword.retrieve(query=query, top_k=max(top_k, 10))
        vector_results = self.vector.retrieve(query=query, top_k=max(top_k, 10))
        fused = reciprocal_rank_fusion([keyword_results, vector_results])
        return fused[:top_k]
