from __future__ import annotations

from pathlib import Path

from rag.ingestion.metadata import infer_category
from rag.types import LoadedDocument


def load_markdown_documents(docs_dir: Path) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []
    for path in sorted(docs_dir.rglob("*.md")):
        documents.append(
            LoadedDocument(
                doc_name=path.name,
                path=str(path.relative_to(docs_dir.parent.parent)),
                category=infer_category(path),
                text=path.read_text(encoding="utf-8").strip(),
            )
        )
    return documents
