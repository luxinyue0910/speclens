from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def get_sentence_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def get_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)
