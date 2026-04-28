from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class LoadedDocument:
    doc_name: str
    path: str
    category: str
    text: str


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    doc_name: str
    path: str
    category: str
    text: str
    chunk_index: int
    source_type: str = "text"
    asset_path: str | None = None
    asset_kind: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    heading_level: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedChunk:
    doc: str
    chunk_id: str
    text: str
    score: float
    category: str
    path: str
    source_type: str = "text"
    asset_path: str | None = None
    asset_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
