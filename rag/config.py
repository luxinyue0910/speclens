from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    docs_dir: Path
    eval_questions_path: Path
    indexes_dir: Path
    chunks_path: Path
    manifest_path: Path
    reports_dir: Path
    asset_cache_dir: Path
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_vision_model: str | None
    image_ocr_enabled: bool
    image_vision_summary_enabled: bool
    tesseract_cmd: str
    ocr_language: str
    default_top_k: int
    default_chunk_size: int
    default_chunk_overlap: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parent.parent
    indexes_dir = root_dir / "data" / "indexes"
    return Settings(
        root_dir=root_dir,
        docs_dir=root_dir / "data" / "docs",
        eval_questions_path=root_dir / "data" / "eval" / "questions.jsonl",
        indexes_dir=indexes_dir,
        chunks_path=indexes_dir / "chunks.jsonl",
        manifest_path=indexes_dir / "manifest.json",
        reports_dir=root_dir / "reports" / "experiments",
        asset_cache_dir=indexes_dir / "assets",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "speclens_docs"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
        ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL") or None,
        image_ocr_enabled=os.getenv("IMAGE_OCR_ENABLED", "true").lower() == "true",
        image_vision_summary_enabled=os.getenv("IMAGE_VISION_SUMMARY_ENABLED", "true").lower()
        == "true",
        tesseract_cmd=os.getenv("TESSERACT_CMD", "tesseract"),
        ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "6")),
        default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "220")),
        default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "40")),
    )


def ensure_runtime_directories(settings: Settings) -> None:
    settings.indexes_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.asset_cache_dir.mkdir(parents=True, exist_ok=True)
