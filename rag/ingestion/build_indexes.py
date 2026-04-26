from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import ensure_runtime_directories, get_settings
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.loader import load_markdown_documents
from rag.resources import get_qdrant_client, get_sentence_encoder
from rag.types import DocumentChunk


def _write_chunks(chunks: list[DocumentChunk], chunks_path) -> None:
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")


def build_indexes(chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict[str, object]:
    settings = get_settings()
    ensure_runtime_directories(settings)

    resolved_chunk_size = chunk_size or settings.default_chunk_size
    resolved_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap

    documents = load_markdown_documents(settings.docs_dir)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_overlap,
    )
    _write_chunks(chunks=chunks, chunks_path=settings.chunks_path)

    encoder = get_sentence_encoder(settings.embedding_model)
    vectors = encoder.encode(
        [chunk.text for chunk in chunks],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    client = get_qdrant_client(settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=encoder.get_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    points = [
        PointStruct(
            id=index,
            vector=vector.tolist(),
            payload=chunk.to_dict(),
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors), start=1)
    ]
    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)

    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_size": resolved_chunk_size,
        "chunk_overlap": resolved_overlap,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
    }
    settings.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "collection_name": settings.qdrant_collection,
        "manifest_path": str(settings.manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SpecLens indexes.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()
    result = build_indexes(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
