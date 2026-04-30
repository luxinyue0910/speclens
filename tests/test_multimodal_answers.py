from __future__ import annotations

import unittest
from pathlib import Path

from rag.config import Settings
from rag.generation.answer_generator import AnswerGenerator
from rag.generation.citation_builder import build_grounded_answer, build_supporting_citations
from rag.types import RetrievedChunk


class MultimodalAnswerTests(unittest.TestCase):
    def _results(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                doc="notification-system-design.md",
                chunk_id="notification-system-design.md#image-1",
                text=(
                    "Image summary: Architecture diagram showing checkout publishing notification jobs to a queue. "
                    "Async workers handle delivery retries outside the request path and send permanently failed jobs to the dead letter topic.\n"
                    "Image text: Async workers retries up to 5 times.\n"
                    "Section: Notification System Design > Architecture.\n"
                    "Image asset: data/assets/diagrams/notification-architecture.svg."
                ),
                score=1.0,
                category="tech_specs",
                path="data/docs/tech_specs/notification-system-design.md",
                source_type="image",
                asset_path="data/assets/diagrams/notification-architecture.svg",
                asset_kind="diagram",
            ),
            RetrievedChunk(
                doc="notification-system-design.md",
                chunk_id="notification-system-design.md#chunk-2",
                text=(
                    "# Notification System Design\n## Architecture\n\n"
                    "- Event producers write notification jobs to a queue\n"
                    "- Workers process jobs asynchronously\n"
                    "- Retries are handled outside the request path"
                ),
                score=0.8,
                category="tech_specs",
                path="data/docs/tech_specs/notification-system-design.md",
            ),
            RetrievedChunk(
                doc="notification-api.md",
                chunk_id="notification-api.md#chunk-3",
                text=(
                    "# Notification API\n## Send notification\n### Notes\n\n"
                    "- `sync_delivery` is deprecated because the platform uses asynchronous workers by default.\n"
                    "- `delivery_mode=async` is the recommended default for all new integrations."
                ),
                score=0.7,
                category="api_docs",
                path="data/docs/api_docs/notification-api.md",
            ),
        ]

    def test_diagram_question_combines_image_summary_and_text_support(self) -> None:
        results = self._results()[:2]

        answer = build_grounded_answer(
            "What does the notification architecture diagram say about retries?",
            results,
        )

        self.assertIn("The architecture diagram shows", answer)
        self.assertIn("retries outside the request path", answer)
        self.assertNotIn("five times", answer)

    def test_visual_questions_prefer_same_doc_text_support_in_citations(self) -> None:
        results = self._results()

        citations = build_supporting_citations(
            question="What does the notification architecture diagram say about retries?",
            answer=build_grounded_answer(
                "What does the notification architecture diagram say about retries?",
                results[:2],
            ),
            results=results,
            limit=2,
        )

        self.assertEqual(citations[0]["chunk_id"], "notification-system-design.md#image-1")
        self.assertEqual(citations[1]["chunk_id"], "notification-system-design.md#chunk-2")

    def test_visual_questions_fall_back_to_grounded_answer_in_generator(self) -> None:
        generator = AnswerGenerator(_build_settings())
        results = self._results()

        normalized = generator._normalize_model_output(
            parsed={
                "answer": (
                    "The architecture diagram indicates that retries for notification jobs "
                    "are handled by Async Workers outside the request path and can occur up "
                    "to five times before failing."
                ),
                "citations": [],
                "confidence": "high",
            },
            question="What does the notification architecture diagram say about retries?",
            results=results,
        )

        self.assertNotIn("five times", normalized["answer"])
        self.assertEqual(
            normalized["citations"][0]["chunk_id"],
            "notification-system-design.md#image-1",
        )


def _build_settings() -> Settings:
    root_dir = Path("/tmp/speclens-test")
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
        ollama_vision_model="gemma3",
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
