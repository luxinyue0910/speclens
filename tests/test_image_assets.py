from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.config import Settings
from rag.ingestion.image_assets import build_image_chunks
from rag.ingestion.ocr import extract_image_text
from rag.types import LoadedDocument


class ImageAssetChunkTests(unittest.TestCase):
    def test_build_image_chunks_uses_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            docs_dir = root_dir / "data" / "docs" / "tech_specs"
            asset_dir = root_dir / "data" / "assets" / "diagrams"
            indexes_dir = root_dir / "data" / "indexes"
            docs_dir.mkdir(parents=True, exist_ok=True)
            asset_dir.mkdir(parents=True, exist_ok=True)
            indexes_dir.mkdir(parents=True, exist_ok=True)

            asset_path = asset_dir / "notification.svg"
            asset_path.write_text("<svg></svg>", encoding="utf-8")
            (asset_dir / "notification.svg.meta.json").write_text(
                json.dumps(
                    {
                        "kind": "diagram",
                        "caption": "Queue-based retries handled by async workers.",
                        "ocr_text": "Async workers retry jobs outside the request path.",
                    }
                ),
                encoding="utf-8",
            )

            document = LoadedDocument(
                doc_name="notification-system-design.md",
                path="data/docs/tech_specs/notification-system-design.md",
                category="tech_specs",
                text=(
                    "# Notification System Design\n\n"
                    "## Architecture\n\n"
                    "![Async architecture](../../assets/diagrams/notification.svg)\n"
                ),
            )

            settings = _build_settings(root_dir)
            chunks = build_image_chunks([document], settings=settings)

            self.assertEqual(len(chunks), 1)
            chunk = chunks[0]
            self.assertEqual(chunk.source_type, "image")
            self.assertEqual(chunk.asset_kind, "diagram")
            self.assertEqual(chunk.asset_path, "data/assets/diagrams/notification.svg")
            self.assertIn("Alt text: Async architecture.", chunk.text)
            self.assertIn("Image summary: Queue-based retries handled by async workers.", chunk.text)
            self.assertIn("Image text: Async workers retry jobs outside the request path.", chunk.text)

    def test_svg_assets_extract_text_without_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            docs_dir = root_dir / "data" / "docs" / "tech_specs"
            asset_dir = root_dir / "data" / "assets" / "diagrams"
            indexes_dir = root_dir / "data" / "indexes"
            docs_dir.mkdir(parents=True, exist_ok=True)
            asset_dir.mkdir(parents=True, exist_ok=True)
            indexes_dir.mkdir(parents=True, exist_ok=True)

            asset_path = asset_dir / "architecture.svg"
            asset_path.write_text(
                (
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    '<text>Notification Queue</text>'
                    '<text>Async Workers</text>'
                    '</svg>'
                ),
                encoding="utf-8",
            )

            settings = _build_settings(root_dir)
            extraction = extract_image_text(asset_path, settings=settings)

            self.assertEqual(extraction.engine, "svg_text")
            self.assertIn("Notification Queue", extraction.text)
            self.assertIn("Async Workers", extraction.text)


def _build_settings(root_dir: Path) -> Settings:
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
        qdrant_url="http://localhost:6333",
        qdrant_collection="speclens_docs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434",
        ollama_model="phi3",
        ollama_vision_model=None,
        image_ocr_enabled=True,
        image_vision_summary_enabled=True,
        tesseract_cmd="tesseract",
        ocr_language="eng",
        default_top_k=6,
        default_chunk_size=220,
        default_chunk_overlap=40,
    )


if __name__ == "__main__":
    unittest.main()
