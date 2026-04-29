from __future__ import annotations

import unittest

from rag.evaluation.metrics import (
    compute_cross_modal_coverage,
    compute_source_hit_rate,
    compute_source_recall,
    has_cross_modal_expectation,
    has_image_expectation,
)


class MultimodalEvalMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_sources = [
            {
                "source_type": "image",
                "doc": "notification-system-design.md",
                "asset_path": "data/assets/diagrams/notification-architecture.svg",
            },
            {
                "source_type": "text",
                "doc": "notification-system-design.md",
            },
        ]

    def test_compute_source_recall_for_image_sources(self) -> None:
        actual_sources = [
            {
                "source_type": "image",
                "doc": "notification-system-design.md",
                "chunk_id": "notification-system-design.md#image-1",
                "asset_path": "data/assets/diagrams/notification-architecture.svg",
            }
        ]

        recall = compute_source_recall(
            expected_sources=self.expected_sources,
            actual_sources=actual_sources,
            source_type="image",
        )

        self.assertEqual(recall, 1.0)

    def test_compute_source_hit_rate_for_image_citations(self) -> None:
        actual_sources = [
            {
                "source_type": "image",
                "doc": "notification-system-design.md",
                "chunk_id": "notification-system-design.md#image-1",
                "asset_path": "data/assets/diagrams/notification-architecture.svg",
            }
        ]

        hit_rate = compute_source_hit_rate(
            expected_sources=self.expected_sources,
            actual_sources=actual_sources,
            source_type="image",
        )

        self.assertEqual(hit_rate, 1.0)

    def test_compute_cross_modal_coverage_requires_text_and_image(self) -> None:
        actual_sources = [
            {
                "source_type": "image",
                "doc": "notification-system-design.md",
                "chunk_id": "notification-system-design.md#image-1",
                "asset_path": "data/assets/diagrams/notification-architecture.svg",
            },
            {
                "source_type": "text",
                "doc": "notification-system-design.md",
                "chunk_id": "notification-system-design.md#chunk-2",
            },
        ]

        coverage = compute_cross_modal_coverage(
            expected_sources=self.expected_sources,
            actual_sources=actual_sources,
        )

        self.assertEqual(coverage, 1.0)

    def test_expectation_helpers_identify_multimodal_questions(self) -> None:
        self.assertTrue(has_image_expectation(self.expected_sources))
        self.assertTrue(has_cross_modal_expectation(self.expected_sources))


if __name__ == "__main__":
    unittest.main()
