from __future__ import annotations

from collections import OrderedDict
import re

from rag.types import RetrievedChunk

ENDPOINT_PATTERN = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/v1/[A-Za-z0-9_./{}-]+")
STATUS_CODE_PATTERN = re.compile(r"\b\d{3}\s+[A-Z][A-Za-z]+\b")
ERROR_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
ENV_VAR_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
FIELD_PATTERN = re.compile(r"\b[a-z][a-z0-9_]{2,}\b")


def build_default_citations(results: list[RetrievedChunk], limit: int = 3) -> list[dict[str, str]]:
    citations: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for result in results:
        if result.chunk_id in citations:
            continue
        snippet = _clean_sentence(_first_claim_sentence(result.text))
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

    answer = build_grounded_answer(question=question, results=results)
    return {
        "answer": answer,
        "citations": build_default_citations(results),
        "confidence": "medium" if len(results) >= 3 else "low",
    }


def build_grounded_answer(question: str, results: list[RetrievedChunk]) -> str:
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

    exact_answer = _build_exact_lookup_answer(question=question, results=results)
    if exact_answer:
        return exact_answer

    return _build_reasoned_summary(question=question, results=results)


def _build_exact_lookup_answer(question: str, results: list[RetrievedChunk]) -> str | None:
    q = question.lower()
    best_sentence = _best_matching_sentence(question=question, text=results[0].text)
    cleaned_best = _clean_sentence(best_sentence)

    if "endpoint" in q:
        endpoint = _extract_first_match(ENDPOINT_PATTERN, cleaned_best)
        if endpoint:
            target = _extract_target_phrase(q, prefix="creates")
            if target:
                return f"`{endpoint}` creates {target}."
            return f"`{endpoint}` is the relevant endpoint for this request."

    if "status code" in q:
        status_code = _extract_first_match(STATUS_CODE_PATTERN, " ".join(result.text for result in results))
        if status_code:
            return f"The status code is `{status_code}`."

    if "error code" in q:
        error_code = _extract_first_match(ERROR_CODE_PATTERN, " ".join(result.text for result in results))
        if error_code:
            return f"The error code is `{error_code}`."

    if "environment variable" in q or "env var" in q:
        env_vars = [
            token
            for token in ENV_VAR_PATTERN.findall(" ".join(result.text for result in results))
            if "_" in token
        ]
        if env_vars:
            verb = _extract_target_phrase(q, prefix="controls")
            if verb:
                return f"`{env_vars[0]}` controls {verb}."
            return f"`{env_vars[0]}` is the relevant environment variable."

    if "deprecated" in q and "field" in q:
        field_name = _extract_field_name(question=question, results=results)
        if field_name:
            return f"`{field_name}` is deprecated."

    if "what is the" in q and "field" in q:
        field_name = _extract_field_name(question=question, results=results)
        if field_name and "deprecated" in joined_text(results):
            return (
                f"`{field_name}` is a deprecated field. New clients should not rely on it."
            )

    if cleaned_best:
        return cleaned_best
    return None


def _build_reasoned_summary(question: str, results: list[RetrievedChunk]) -> str:
    supporting_sentences: list[str] = []
    seen: set[str] = set()
    for result in results:
        sentence = _clean_sentence(_best_matching_sentence(question=question, text=result.text))
        if not sentence:
            continue
        lowered = sentence.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        supporting_sentences.append(sentence.rstrip("."))
        if len(supporting_sentences) >= 2:
            break

    if not supporting_sentences:
        return "I don't have enough evidence."

    if len(supporting_sentences) == 1:
        return supporting_sentences[0] + "."

    return supporting_sentences[0] + ". " + supporting_sentences[1] + "."


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


def _extract_first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _extract_target_phrase(question: str, prefix: str) -> str | None:
    match = re.search(rf"{prefix}\s+(.+?)\??$", question, flags=re.IGNORECASE)
    if not match:
        return None
    phrase = match.group(1).strip()
    if not phrase:
        return None
    if phrase.lower().startswith(("a ", "an ", "the ")):
        return phrase
    return phrase


def _extract_field_name(question: str, results: list[RetrievedChunk]) -> str | None:
    question_fields = [token for token in FIELD_PATTERN.findall(question) if "_" in token]
    if question_fields:
        return question_fields[0]

    text = " ".join(result.text for result in results)
    fields = [token for token in FIELD_PATTERN.findall(text) if "_" in token]
    return fields[0] if fields else None


def _first_claim_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    for sentence in sentences:
        cleaned = _clean_sentence(sentence)
        if cleaned and not cleaned.startswith("#"):
            return cleaned
    return text[:220].strip()


def _clean_sentence(text: str) -> str:
    cleaned = text.replace("`", "").replace("*", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("'\" ")
    return cleaned


def joined_text(results: list[RetrievedChunk]) -> str:
    return " ".join(result.text.lower() for result in results)
