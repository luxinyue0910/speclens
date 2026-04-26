from __future__ import annotations


def compute_retrieval_recall(expected_docs: list[str], retrieved_docs: list[str]) -> float:
    if not expected_docs:
        return 0.0
    expected = set(expected_docs)
    retrieved = set(retrieved_docs)
    return len(expected & retrieved) / len(expected)


def compute_expected_doc_hit_rate(expected_docs: list[str], retrieved_docs: list[str]) -> float:
    expected = set(expected_docs)
    retrieved = set(retrieved_docs)
    return 1.0 if expected & retrieved else 0.0
