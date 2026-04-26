from __future__ import annotations

import json

import requests

from rag.config import Settings
from rag.generation.citation_builder import build_default_citations, build_extractive_answer
from rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.types import RetrievedChunk


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
                "format": "json",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "").strip()
        parsed = json.loads(raw)

        citations = parsed.get("citations") or build_default_citations(results)
        return {
            "answer": parsed.get("answer", "I don't have enough evidence."),
            "citations": citations,
            "confidence": parsed.get("confidence", "medium"),
        }
