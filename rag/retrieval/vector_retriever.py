from __future__ import annotations

from rag.config import Settings
from rag.resources import get_qdrant_client, get_sentence_encoder
from rag.types import RetrievedChunk


class VectorRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = get_qdrant_client(settings.qdrant_url)
        self.encoder = get_sentence_encoder(settings.embedding_model)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        points = self._query_points(vector=vector, top_k=top_k)
        results: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    doc=str(payload["doc_name"]),
                    chunk_id=str(payload["chunk_id"]),
                    text=str(payload["text"]),
                    score=float(point.score),
                    category=str(payload["category"]),
                    path=str(payload["path"]),
                    source_type=str(payload.get("source_type", "text")),
                    asset_path=str(payload["asset_path"]) if payload.get("asset_path") else None,
                    asset_kind=str(payload["asset_kind"]) if payload.get("asset_kind") else None,
                )
            )
        return results

    def _query_points(self, vector: list[float], top_k: int):
        try:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        except AttributeError:
            return self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
