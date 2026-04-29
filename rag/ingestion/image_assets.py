from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.config import Settings
from rag.ingestion.metadata import build_image_chunk_id
from rag.ingestion.ocr import extract_image_text
from rag.ingestion.vision_summary import generate_image_summary
from rag.types import DocumentChunk, LoadedDocument


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


@dataclass(slots=True)
class ImageAssetMetadata:
    caption: str = ""
    ocr_text: str = ""
    kind: str = "image"


def build_image_chunks(
    documents: list[LoadedDocument],
    settings: Settings,
) -> list[DocumentChunk]:
    resolved_root_dir = settings.root_dir.resolve()
    chunks: list[DocumentChunk] = []

    for document in documents:
        heading_stack: list[tuple[int, str]] = []
        image_index = 0
        doc_path = settings.root_dir / document.path

        for line in document.text.splitlines():
            heading_match = HEADING_PATTERN.match(line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = [item for item in heading_stack if item[0] < level]
                heading_stack.append((level, title))
                continue

            image_match = IMAGE_PATTERN.search(line.strip())
            if not image_match:
                continue

            image_index += 1
            raw_asset_path = image_match.group("path").strip()
            alt_text = image_match.group("alt").strip()
            resolved_asset_path = _resolve_asset_path(doc_path=doc_path, raw_asset_path=raw_asset_path)
            sidecar = _load_asset_metadata(resolved_asset_path)
            ocr = extract_image_text(resolved_asset_path, settings=settings)
            vision = generate_image_summary(resolved_asset_path, settings=settings)
            metadata = _merge_asset_metadata(sidecar=sidecar, ocr_text=ocr.text, vision_summary=vision.summary)
            relative_asset_path = str(resolved_asset_path.relative_to(resolved_root_dir))
            section_path = " > ".join(title for _, title in heading_stack) or None
            section_title = heading_stack[-1][1] if heading_stack else None
            heading_level = heading_stack[-1][0] if heading_stack else None

            chunk_text = _build_image_chunk_text(
                section_path=section_path,
                alt_text=alt_text,
                asset_path=relative_asset_path,
                metadata=metadata,
                ocr_engine=ocr.engine,
                vision_engine=vision.engine,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=build_image_chunk_id(document.doc_name, image_index),
                    doc_name=document.doc_name,
                    path=document.path,
                    category=document.category,
                    text=chunk_text,
                    chunk_index=image_index,
                    source_type="image",
                    asset_path=relative_asset_path,
                    asset_kind=metadata.kind,
                    section_title=section_title,
                    section_path=section_path,
                    heading_level=heading_level,
                )
            )

    return chunks


def _resolve_asset_path(doc_path: Path, raw_asset_path: str) -> Path:
    clean_path = raw_asset_path.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    return (doc_path.parent / clean_path).resolve()


def _load_asset_metadata(asset_path: Path) -> ImageAssetMetadata:
    candidates = [
        asset_path.with_suffix(asset_path.suffix + ".meta.json"),
        asset_path.with_suffix(".meta.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return ImageAssetMetadata(
                caption=str(payload.get("caption", "")).strip(),
                ocr_text=str(payload.get("ocr_text", "")).strip(),
                kind=str(payload.get("kind", "image")).strip() or "image",
            )
    return ImageAssetMetadata(kind=asset_path.suffix.lstrip(".") or "image")


def _merge_asset_metadata(
    sidecar: ImageAssetMetadata,
    ocr_text: str,
    vision_summary: str,
) -> ImageAssetMetadata:
    return ImageAssetMetadata(
        caption=sidecar.caption or vision_summary,
        ocr_text=sidecar.ocr_text or ocr_text,
        kind=sidecar.kind,
    )


def _build_image_chunk_text(
    section_path: str | None,
    alt_text: str,
    asset_path: str,
    metadata: ImageAssetMetadata,
    ocr_engine: str,
    vision_engine: str,
) -> str:
    lines = []
    if metadata.caption:
        lines.append(_ensure_period(f"Image summary: {metadata.caption}"))
    if metadata.ocr_text:
        lines.append(_ensure_period(f"Image text: {metadata.ocr_text}"))
    if alt_text:
        lines.append(_ensure_period(f"Alt text: {alt_text}"))
    if section_path:
        lines.append(_ensure_period(f"Section: {section_path}"))
    lines.append(_ensure_period(f"Image asset: {asset_path}"))
    if vision_engine and vision_engine != "none":
        lines.append(_ensure_period(f"Vision engine: {vision_engine}"))
    if ocr_engine and ocr_engine != "none":
        lines.append(_ensure_period(f"OCR engine: {ocr_engine}"))
    return "\n".join(lines)


def _ensure_period(text: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return cleaned + "."
