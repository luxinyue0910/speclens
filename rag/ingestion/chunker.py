from __future__ import annotations

import re

from rag.ingestion.metadata import build_chunk_id
from rag.types import DocumentChunk, LoadedDocument


def chunk_document(
    document: LoadedDocument, chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    words = re.split(r"\s+", document.text.strip())
    if not words or words == [""]:
        return []

    chunks: list[DocumentChunk] = []
    step = max(1, chunk_size - chunk_overlap)
    chunk_index = 0

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            continue
        chunk_index += 1
        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(document.doc_name, chunk_index),
                doc_name=document.doc_name,
                path=document.path,
                category=document.category,
                text=" ".join(window).strip(),
                chunk_index=chunk_index,
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: list[LoadedDocument], chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks
