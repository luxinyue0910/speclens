from __future__ import annotations

import re
from dataclasses import dataclass

from rag.ingestion.metadata import build_chunk_id
from rag.types import DocumentChunk, LoadedDocument


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class MarkdownSection:
    headings: tuple[tuple[int, str], ...]
    content: str

    @property
    def section_title(self) -> str | None:
        if not self.headings:
            return None
        return self.headings[-1][1]

    @property
    def section_path(self) -> str | None:
        if not self.headings:
            return None
        return " > ".join(title for _, title in self.headings)

    @property
    def heading_level(self) -> int | None:
        if not self.headings:
            return None
        return self.headings[-1][0]


def chunk_document(
    document: LoadedDocument, chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    sections = _split_markdown_sections(document.text)
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in sections:
        body_words = _tokenize_words(section.content)
        if not body_words:
            continue

        heading_context = _render_heading_context(section.headings)
        heading_words = _tokenize_words(heading_context)
        available_body_size = max(1, chunk_size - len(heading_words))
        step = max(1, available_body_size - chunk_overlap)

        for start in range(0, len(body_words), step):
            window = body_words[start : start + available_body_size]
            if not window:
                continue

            chunk_index += 1
            chunk_text = _compose_chunk_text(heading_context=heading_context, body_words=window)
            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(document.doc_name, chunk_index),
                    doc_name=document.doc_name,
                    path=document.path,
                    category=document.category,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    section_title=section.section_title,
                    section_path=section.section_path,
                    heading_level=section.heading_level,
                )
            )
            if start + available_body_size >= len(body_words):
                break

    return chunks


def chunk_documents(
    documents: list[LoadedDocument], chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _split_markdown_sections(text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_headings: tuple[tuple[int, str], ...] = ()
    current_content: list[str] = []

    for line in text.splitlines():
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            _append_section(
                sections=sections,
                headings=current_headings,
                content_lines=current_content,
            )

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            current_headings = tuple(heading_stack)
            current_content = []
            continue

        current_content.append(line)

    _append_section(
        sections=sections,
        headings=current_headings,
        content_lines=current_content,
    )

    if sections:
        return sections

    fallback_content = text.strip()
    if not fallback_content:
        return []
    return [MarkdownSection(headings=(), content=fallback_content)]


def _append_section(
    sections: list[MarkdownSection],
    headings: tuple[tuple[int, str], ...],
    content_lines: list[str],
) -> None:
    content = "\n".join(content_lines).strip()
    if not content:
        return
    sections.append(MarkdownSection(headings=headings, content=content))


def _render_heading_context(headings: tuple[tuple[int, str], ...]) -> str:
    if not headings:
        return ""
    return "\n".join(f"{'#' * level} {title}" for level, title in headings)


def _compose_chunk_text(heading_context: str, body_words: list[str]) -> str:
    body = " ".join(body_words).strip()
    if heading_context and body:
        return f"{heading_context}\n\n{body}"
    return heading_context or body


def _tokenize_words(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    return [word for word in WHITESPACE_PATTERN.split(normalized) if word]
