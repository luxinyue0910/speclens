from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.ingestion.image_assets import build_image_chunks
from rag.types import LoadedDocument


class ImageAssetChunkTests(unittest.TestCase):
    def test_build_image_chunks_uses_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            docs_dir = root_dir / "data" / "docs" / "tech_specs"
            asset_dir = root_dir / "data" / "assets" / "diagrams"
            docs_dir.mkdir(parents=True, exist_ok=True)
            asset_dir.mkdir(parents=True, exist_ok=True)

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

            chunks = build_image_chunks([document], root_dir=root_dir)

            self.assertEqual(len(chunks), 1)
            chunk = chunks[0]
            self.assertEqual(chunk.source_type, "image")
            self.assertEqual(chunk.asset_kind, "diagram")
            self.assertEqual(chunk.asset_path, "data/assets/diagrams/notification.svg")
            self.assertIn("Alt text: Async architecture", chunk.text)
            self.assertIn("Image summary: Queue-based retries handled by async workers.", chunk.text)
            self.assertIn("Image text: Async workers retry jobs outside the request path.", chunk.text)


if __name__ == "__main__":
    unittest.main()
