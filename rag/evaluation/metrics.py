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


def compute_source_recall(
    expected_sources: list[dict[str, object]],
    actual_sources: list[dict[str, object]],
    source_type: str | None = None,
) -> float:
    filtered_expected = _filter_expected_sources(expected_sources, source_type=source_type)
    if not filtered_expected:
        return 0.0

    matched = 0
    for expected in filtered_expected:
        if any(_source_matches(expected, actual) for actual in actual_sources):
            matched += 1
    return matched / len(filtered_expected)


def compute_source_hit_rate(
    expected_sources: list[dict[str, object]],
    actual_sources: list[dict[str, object]],
    source_type: str | None = None,
) -> float:
    filtered_expected = _filter_expected_sources(expected_sources, source_type=source_type)
    if not filtered_expected:
        return 0.0
    return 1.0 if compute_source_recall(filtered_expected, actual_sources) > 0 else 0.0


def compute_cross_modal_coverage(
    expected_sources: list[dict[str, object]],
    actual_sources: list[dict[str, object]],
) -> float:
    expected_types = {
        str(item.get("source_type", "")).strip()
        for item in expected_sources
        if item.get("source_type")
    }
    if not {"text", "image"}.issubset(expected_types):
        return 0.0

    for required_type in ("text", "image"):
        required_sources = _filter_expected_sources(expected_sources, source_type=required_type)
        if not any(
            any(_source_matches(expected, actual) for actual in actual_sources)
            for expected in required_sources
        ):
            return 0.0
    return 1.0


def has_image_expectation(expected_sources: list[dict[str, object]]) -> bool:
    return any(item.get("source_type") == "image" for item in expected_sources)


def has_cross_modal_expectation(expected_sources: list[dict[str, object]]) -> bool:
    expected_types = {
        str(item.get("source_type", "")).strip()
        for item in expected_sources
        if item.get("source_type")
    }
    return {"text", "image"}.issubset(expected_types)


def _filter_expected_sources(
    expected_sources: list[dict[str, object]],
    source_type: str | None = None,
) -> list[dict[str, object]]:
    if source_type is None:
        return list(expected_sources)
    return [item for item in expected_sources if item.get("source_type") == source_type]


def _source_matches(expected: dict[str, object], actual: dict[str, object]) -> bool:
    expected_source_type = str(expected.get("source_type", "")).strip()
    actual_source_type = str(actual.get("source_type", "text")).strip()
    if expected_source_type and actual_source_type != expected_source_type:
        return False

    for key in ("asset_path", "doc", "chunk_id"):
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        actual_value = actual.get(key)
        if actual_value != expected_value:
            return False
    return True
