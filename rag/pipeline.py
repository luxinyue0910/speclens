from __future__ import annotations

from time import perf_counter

from rag.config import Settings
from rag.generation.answer_generator import AnswerGenerator
from rag.ingestion.build_indexes import build_indexes
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.keyword_retriever import KeywordRetriever
from rag.retrieval.reranker import maybe_rerank
from rag.retrieval.vector_retriever import VectorRetriever
from rag.routing.query_router import route_query


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    def ingest(self, chunk_size: int, chunk_overlap: int) -> dict[str, object]:
        return build_indexes(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ask(
        self,
        question: str,
        retriever: str = "auto",
        top_k: int | None = None,
        generator: str = "auto",
    ) -> dict[str, object]:
        started = perf_counter()
        selected_route = route_query(question) if retriever == "auto" else retriever
        resolved_top_k = top_k or self.settings.default_top_k
        results = self._retrieve(selected_route, question=question, top_k=resolved_top_k)
        results = maybe_rerank(results, enabled=False)

        generated = AnswerGenerator(self.settings).generate(
            question=question,
            results=results,
            generator=generator,
        )
        latency_ms = int((perf_counter() - started) * 1000)

        return {
            "question": question,
            "route": selected_route,
            "answer": generated["answer"],
            "citations": generated["citations"],
            "retrieved_chunks": [result.to_dict() for result in results],
            "latency_ms": latency_ms,
            "confidence": generated["confidence"],
        }

    def _retrieve(self, route: str, question: str, top_k: int):
        if route == "keyword":
            return KeywordRetriever(self.settings).retrieve(question, top_k)
        if route == "vector":
            return VectorRetriever(self.settings).retrieve(question, top_k)
        return HybridRetriever(self.settings).retrieve(question, top_k)
