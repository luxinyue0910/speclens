from __future__ import annotations

import unittest

from rag.ingestion.chunker import chunk_document
from rag.types import LoadedDocument


class ChunkerTests(unittest.TestCase):
    def test_chunk_document_tracks_section_metadata(self) -> None:
        document = LoadedDocument(
            doc_name="payment-api.md",
            path="data/docs/api_docs/payment-api.md",
            category="api_docs",
            text=(
                "# Payment API\n\n"
                "## Create payment intent\n\n"
                "`POST /v1/payment-intents`\n\n"
                "### Notes\n\n"
                "- `refund_mode` is deprecated.\n"
            ),
        )

        chunks = chunk_document(document, chunk_size=40, chunk_overlap=5)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_title, "Create payment intent")
        self.assertEqual(chunks[0].section_path, "Payment API > Create payment intent")
        self.assertIn("## Create payment intent", chunks[0].text)
        self.assertEqual(chunks[1].section_title, "Notes")
        self.assertEqual(chunks[1].section_path, "Payment API > Create payment intent > Notes")
        self.assertIn("### Notes", chunks[1].text)

    def test_long_section_keeps_heading_context_on_followup_chunks(self) -> None:
        document = LoadedDocument(
            doc_name="long-doc.md",
            path="data/docs/tech_specs/long-doc.md",
            category="tech_specs",
            text=(
                "# Search Design\n\n"
                "## Ranking rationale\n\n"
                + " ".join(f"token{i}" for i in range(60))
            ),
        )

        chunks = chunk_document(document, chunk_size=18, chunk_overlap=3)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.section_title, "Ranking rationale")
            self.assertIn("# Search Design", chunk.text)
            self.assertIn("## Ranking rationale", chunk.text)


if __name__ == "__main__":
    unittest.main()
