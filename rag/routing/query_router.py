from __future__ import annotations


EXACT_TERMS = {
    "endpoint",
    "field",
    "parameter",
    "status code",
    "deprecated",
    "config",
    "env var",
    "environment variable",
    "error code",
    "header",
}

REASONING_TERMS = {
    "why",
    "tradeoff",
    "root cause",
    "compare",
    "difference",
    "impact",
    "decision",
    "rationale",
    "how do",
    "how does",
    "how did",
    "explain",
    "respond",
}

CONFLICT_TERMS = {
    "support",
    "does",
    "available",
    "conflict",
    "contradict",
    "inconsistent",
    "deprecated",
    "by default",
    "within the latency goal",
    "triggered directly",
    "mainly caused",
}


def route_query(query: str) -> str:
    normalized = query.lower()

    if any(term in normalized for term in EXACT_TERMS):
        return "keyword"

    if any(term in normalized for term in REASONING_TERMS):
        return "hybrid"

    if any(term in normalized for term in CONFLICT_TERMS):
        return "hybrid"

    return "vector"
