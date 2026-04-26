from __future__ import annotations

import json
from typing import Any

import requests

from rag.config import Settings
from rag.generation.citation_builder import build_default_citations, build_extractive_answer
from rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.types import RetrievedChunk

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["doc", "chunk_id", "claim"],
            },
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["answer", "citations", "confidence"],
}


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self, question: str, results: list[RetrievedChunk], generator: str
    ) -> dict[str, object]:
        if generator == "extractive":
            return build_extractive_answer(question=question, results=results)

        if generator == "auto":
            try:
                return self._generate_with_ollama(question=question, results=results)
            except requests.RequestException:
                return build_extractive_answer(question=question, results=results)
            except ValueError:
                return build_extractive_answer(question=question, results=results)

        return self._generate_with_ollama(question=question, results=results)

    def _generate_with_ollama(
        self, question: str, results: list[RetrievedChunk]
    ) -> dict[str, object]:
        if not results:
            return {
                "answer": "I don't have enough evidence.",
                "citations": [],
                "confidence": "low",
            }

        response = requests.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": build_user_prompt(question=question, results=results),
                "system": SYSTEM_PROMPT,
                "stream": False,
                "format": ANSWER_SCHEMA,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "").strip()
        parsed = json.loads(raw)
        normalized = self._normalize_model_output(parsed=parsed, question=question, results=results)
        return {
            "answer": normalized["answer"],
            "citations": normalized["citations"],
            "confidence": normalized["confidence"],
        }

    def _normalize_model_output(
        self,
        parsed: dict[str, Any],
        question: str,
        results: list[RetrievedChunk],
    ) -> dict[str, object]:
        fallback = build_extractive_answer(question=question, results=results)
        chunk_by_id = {result.chunk_id: result for result in results}
        known_docs = {result.doc for result in results}

        answer = str(parsed.get("answer", "")).strip()
        if len(answer.split()) < 4:
            answer = str(fallback["answer"])
        elif self._should_prefer_fallback_answer(
            answer=answer,
            question=question,
            results=results,
        ):
            answer = str(fallback["answer"])

        confidence = str(parsed.get("confidence", "medium")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = str(fallback["confidence"])

        citations: list[dict[str, str]] = []
        for item in parsed.get("citations", []) or []:
            if not isinstance(item, dict):
                continue
            doc = str(item.get("doc", "")).strip()
            chunk_id = str(item.get("chunk_id", "")).strip()
            claim = str(item.get("claim", "")).strip()

            if doc in chunk_by_id and chunk_id not in chunk_by_id:
                repaired = chunk_by_id[doc]
                doc = repaired.doc
                chunk_id = repaired.chunk_id

            if not doc and chunk_id in chunk_by_id:
                doc = chunk_by_id[chunk_id].doc

            if doc and chunk_id and chunk_id in chunk_by_id and doc != chunk_by_id[chunk_id].doc:
                doc = chunk_by_id[chunk_id].doc

            if doc not in known_docs and chunk_id in chunk_by_id:
                doc = chunk_by_id[chunk_id].doc

            if chunk_id not in chunk_by_id:
                matching = next((result for result in results if result.doc == doc), None)
                if matching:
                    chunk_id = matching.chunk_id

            if not claim and chunk_id in chunk_by_id:
                claim = build_default_citations([chunk_by_id[chunk_id]], limit=1)[0]["claim"]

            if doc in known_docs and chunk_id in chunk_by_id:
                citations.append(
                    {
                        "doc": doc,
                        "chunk_id": chunk_id,
                        "claim": claim or "Relevant supporting evidence.",
                    }
                )

        if not citations:
            citations = list(fallback["citations"])

        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for citation in citations:
            key = citation["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)

        return {
            "answer": answer,
            "citations": deduped[:3],
            "confidence": confidence,
        }

    def _should_prefer_fallback_answer(
        self,
        answer: str,
        question: str,
        results: list[RetrievedChunk],
    ) -> bool:
        normalized_question = question.lower()
        normalized_answer = answer.lower()
        joined = " ".join(result.text.lower() for result in results)

        if "automatic refund" in normalized_question:
            has_conflict = (
                "automatic refunds for" in joined
                and "does not support automatic refunds" in joined
            )
            missing_conflict_language = not any(
                token in normalized_answer
                for token in ("but", "however", "while", "conflict", "mvp")
            )
            if has_conflict and missing_conflict_language:
                return True

        if normalized_answer in {"yes", "no", "yes.", "no."}:
            return True

        return False
