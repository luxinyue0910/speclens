from __future__ import annotations

import unittest

from rag.generation.citation_builder import build_grounded_answer
from rag.types import RetrievedChunk


class MultimodalAnswerTests(unittest.TestCase):
    def test_diagram_question_combines_image_summary_and_text_support(self) -> None:
        results = [
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
        ]

        answer = build_grounded_answer(
            "What does the notification architecture diagram say about retries?",
            results,
        )

        self.assertIn("The architecture diagram shows", answer)
        self.assertIn("retries outside the request path", answer)


if __name__ == "__main__":
    unittest.main()
