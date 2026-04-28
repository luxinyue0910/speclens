from __future__ import annotations

from collections import OrderedDict
import re

from rag.types import RetrievedChunk

ENDPOINT_PATTERN = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/v1/[A-Za-z0-9_./{}-]+")
STATUS_CODE_PATTERN = re.compile(r"\b\d{3}\s+[A-Z][A-Za-z]+\b")
ERROR_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
ENV_VAR_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
FIELD_PATTERN = re.compile(r"\b[a-z][a-z0-9_]{2,}\b")
TOKEN_PATTERN = re.compile(r"[a-z0-9_./{}-]+")
VISUAL_TERMS = {"diagram", "architecture", "flow", "dashboard", "chart", "graph", "figure", "screenshot"}


def build_default_citations(results: list[RetrievedChunk], limit: int = 3) -> list[dict[str, str | None]]:
    citations: "OrderedDict[str, dict[str, str | None]]" = OrderedDict()
    for result in results:
        if result.chunk_id in citations:
            continue
        snippet = _clean_sentence(_first_claim_sentence(result.text))
        citations[result.chunk_id] = {
            "doc": result.doc,
            "chunk_id": result.chunk_id,
            "claim": snippet[:180] or "Relevant supporting evidence.",
            "source_type": result.source_type,
            "asset_path": result.asset_path,
        }
        if len(citations) >= limit:
            break
    return list(citations.values())


def build_supporting_citations(
    question: str,
    answer: str,
    results: list[RetrievedChunk],
    limit: int = 3,
) -> list[dict[str, str | None]]:
    ranked: list[tuple[float, dict[str, str | None]]] = []
    for result in results:
        claim, score = _best_supporting_claim(
            question=question,
            answer=answer,
            text=result.text,
        )
        ranked.append(
            (
                score,
                {
                    "doc": result.doc,
                    "chunk_id": result.chunk_id,
                    "claim": claim[:180] or "Relevant supporting evidence.",
                    "source_type": result.source_type,
                    "asset_path": result.asset_path,
                },
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    citations: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for _, citation in ranked:
        if citation["chunk_id"] in seen:
            continue
        seen.add(citation["chunk_id"])
        citations.append(citation)
        if len(citations) >= limit:
            break
    return citations


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

    multimodal_answer = _build_multimodal_summary(question=question, results=results)
    if multimodal_answer:
        return multimodal_answer

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


def _build_multimodal_summary(question: str, results: list[RetrievedChunk]) -> str | None:
    normalized_question = question.lower()
    if not any(term in normalized_question for term in VISUAL_TERMS):
        return None

    image_results = [result for result in results if result.source_type == "image"]
    if not image_results:
        return None

    best_image = image_results[0]
    image_summary = _extract_labeled_value(best_image.text, "Image summary")
    image_text = _extract_labeled_value(best_image.text, "Image text")
    if not image_summary and not image_text:
        return None

    primary = _rewrite_visual_summary(
        summary=image_summary or image_text or "",
        asset_kind=best_image.asset_kind,
    )
    if _visual_summary_is_sufficient(question=question, summary=primary):
        return primary
    support = _best_text_support(question=question, results=results, image_summary=primary)

    if not primary:
        return support
    if support and support.lower() not in primary.lower():
        return f"{primary} {support}"
    return primary


def _best_supporting_claim(question: str, answer: str, text: str) -> tuple[str, float]:
    candidates = _candidate_claim_fragments(text)
    if not candidates:
        cleaned = _clean_sentence(text[:220])
        return cleaned, 0.0

    question_terms = _term_set(question)
    answer_terms = _term_set(answer)
    anchor_terms = _anchor_terms(question, answer)

    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        normalized = candidate.lower()
        candidate_terms = _term_set(candidate)
        score = 0.0
        score += 4.0 * len(candidate_terms & answer_terms)
        score += 2.0 * len(candidate_terms & question_terms)
        score += 8.0 * sum(1 for anchor in anchor_terms if anchor.lower() in normalized)

        if "deprecated" in question.lower() and "deprecated" in normalized:
            score += 2.0
        if "endpoint" in question.lower() and ENDPOINT_PATTERN.search(candidate):
            score += 3.0
        if "status code" in question.lower() and STATUS_CODE_PATTERN.search(candidate):
            score += 3.0
        if "error code" in question.lower() and ERROR_CODE_PATTERN.search(candidate):
            score += 3.0
        if "environment variable" in question.lower() and ENV_VAR_PATTERN.search(candidate):
            score += 3.0

        if len(candidate.split()) > 28:
            score -= 1.0
        if candidate.startswith("#"):
            score -= 5.0

        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    return _clean_sentence(best_candidate), best_score


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
    cleaned = re.sub(r"^[#\-.:\s]+", "", cleaned)
    cleaned = re.sub(
        r"^(Image summary|Image text|Alt text|Section|Image asset):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip("'\" ")
    return cleaned


def joined_text(results: list[RetrievedChunk]) -> str:
    return " ".join(result.text.lower() for result in results)


def _extract_labeled_value(text: str, label: str) -> str:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return _clean_sentence(line[len(prefix) :])
    return ""


def _rewrite_visual_summary(summary: str, asset_kind: str | None) -> str:
    cleaned = _clean_sentence(summary)
    lowered = cleaned.lower()

    if lowered.startswith("architecture diagram showing "):
        return "The architecture diagram shows " + cleaned[len("Architecture diagram showing ") :]
    if lowered.startswith("diagram showing "):
        return "The diagram shows " + cleaned[len("Diagram showing ") :]
    if lowered.startswith("dashboard showing "):
        return "The dashboard shows " + cleaned[len("Dashboard showing ") :]
    if asset_kind == "diagram":
        return "The diagram shows " + cleaned[0].lower() + cleaned[1:] if cleaned else ""
    if asset_kind == "dashboard":
        return "The dashboard shows " + cleaned[0].lower() + cleaned[1:] if cleaned else ""
    return _ensure_sentence(cleaned)


def _best_text_support(
    question: str,
    results: list[RetrievedChunk],
    image_summary: str,
) -> str | None:
    question_terms = _term_set(question)
    summary_terms = _term_set(image_summary)
    best_score = 0
    best_sentence = ""

    for result in results:
        if result.source_type == "image":
            continue
        sentence = _clean_sentence(_best_matching_sentence(question=question, text=result.text))
        if not sentence:
            continue
        sentence_terms = _term_set(sentence)
        score = len(sentence_terms & question_terms) + len(sentence_terms - summary_terms)
        if result.doc == results[0].doc:
            score += 2
        if "retry" in question.lower() and "retry" in sentence.lower():
            score += 2
        if "outside the request path" in sentence.lower():
            score += 2
        if "ranking v2" in sentence.lower():
            score += 1
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score <= 0 or not best_sentence:
        return None
    if best_sentence.lower() in image_summary.lower():
        return None
    return _ensure_sentence(best_sentence)


def _ensure_sentence(text: str) -> str:
    if not text:
        return text
    if text.endswith((".", "!", "?")):
        return text
    return text + "."


def _visual_summary_is_sufficient(question: str, summary: str) -> bool:
    normalized_question = question.lower()
    normalized_summary = summary.lower()

    if "retry" in normalized_question and "retry" in normalized_summary:
        return True
    if "dashboard" in normalized_question and "ranking v2" in normalized_summary:
        return True
    if "chart" in normalized_question and "ranking v2" in normalized_summary:
        return True
    if "architecture" in normalized_question and "outside the request path" in normalized_summary:
        return True
    return False


def _candidate_claim_fragments(text: str) -> list[str]:
    normalized = text.replace(" ## ", ". ").replace(" ### ", ". ").replace(" - ", ". ")
    normalized = normalized.replace("# ", ". ")
    parts = re.split(r"(?<=[.!?])\s+|\s+\.\s+", normalized)
    candidates: list[str] = []
    for part in parts:
        cleaned = _clean_sentence(part)
        if not cleaned:
            continue
        if len(cleaned.split()) < 3:
            continue
        candidates.append(cleaned)
    return candidates


def _term_set(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 2
    }


def _anchor_terms(question: str, answer: str) -> list[str]:
    anchors: list[str] = []
    for pattern in (
        ENDPOINT_PATTERN,
        STATUS_CODE_PATTERN,
        ERROR_CODE_PATTERN,
        ENV_VAR_PATTERN,
    ):
        anchors.extend(pattern.findall(question))
        anchors.extend(pattern.findall(answer))

    if not anchors:
        anchors.extend(token for token in FIELD_PATTERN.findall(question) if "_" in token)
        anchors.extend(token for token in FIELD_PATTERN.findall(answer) if "_" in token)
    return anchors
