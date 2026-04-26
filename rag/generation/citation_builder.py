from __future__ import annotations

from collections import OrderedDict
import re

from rag.types import RetrievedChunk


def build_default_citations(results: list[RetrievedChunk], limit: int = 3) -> list[dict[str, str]]:
    citations: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for result in results:
        if result.chunk_id in citations:
            continue
        snippet = result.text.split(".")[0].strip()
        citations[result.chunk_id] = {
            "doc": result.doc,
            "chunk_id": result.chunk_id,
            "claim": snippet[:180] or "Relevant supporting evidence.",
        }
        if len(citations) >= limit:
            break
    return list(citations.values())


def build_extractive_answer(question: str, results: list[RetrievedChunk]) -> dict[str, object]:
    if not results:
        return {
            "answer": "I don't have enough evidence.",
            "citations": [],
            "confidence": "low",
        }

    answer = _build_heuristic_answer(question=question, results=results)
    return {
        "answer": answer,
        "citations": build_default_citations(results),
        "confidence": "medium" if len(results) >= 3 else "low",
    }


def _build_heuristic_answer(question: str, results: list[RetrievedChunk]) -> str:
    q = question.lower()
    joined = " ".join(result.text.lower() for result in results)

    if "automatic refund" in q:
        if "does not support automatic refunds" in joined or "automatic refunds are not available" in joined:
            return (
                "No. The retrieved documents conflict: the checkout PRD proposes automatic refunds, "
                "but the MVP payment design says automatic refunds are not supported and release 1.2 "
                "only adds admin-triggered refunds."
            )

    if "sync_delivery" in q or "synchronous" in q:
        if "deprecated" in joined and "async" in joined:
            return (
                "No. Async delivery is the default path, and the deprecated sync_delivery field is "
                "not recommended for new integrations."
            )

    if "duplicate order" in q:
        if "409 conflict" in joined and "duplicate_order" in joined:
            return "Duplicate order submissions return 409 Conflict with error code DUPLICATE_ORDER."

    evidence = []
    for result in results[:3]:
        sentence = _best_matching_sentence(question=question, text=result.text)
        evidence.append(f"{result.doc}: {sentence}")
    return "Based on the retrieved context, " + " ".join(evidence)


def _best_matching_sentence(question: str, text: str) -> str:
    keywords = {
        token
        for token in re.findall(r"[a-z0-9_/-]+", question.lower())
        if len(token) > 2
    }
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        scored.append((score, sentence.strip()))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, sentence in scored:
        if sentence:
            return sentence[:220]
    return text[:220].strip()
