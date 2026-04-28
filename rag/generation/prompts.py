from __future__ import annotations

from rag.types import RetrievedChunk


SYSTEM_PROMPT = """You are an engineering documentation assistant.

Answer only using the retrieved context.
If the context is insufficient, say "I don't have enough evidence."
If documents conflict, explicitly mention the conflict.
Every key claim must cite source documents.
Use concise engineering language.
Write a complete answer of 2 to 4 sentences, not a one-word answer.
Return valid JSON with keys: answer, citations, confidence.
Each citation must include: doc, chunk_id, claim.
If a retrieved item comes from an image asset, you may also reference its asset_path.
"""


def build_context_block(results: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        prefix = f"[{index}] doc={result.doc} chunk_id={result.chunk_id}"
        if result.source_type != "text":
            prefix += f" source_type={result.source_type}"
        if result.asset_path:
            prefix += f" asset_path={result.asset_path}"
        lines.append(
            f"{prefix}\n{result.text}"
        )
    return "\n\n".join(lines)


def build_user_prompt(question: str, results: list[RetrievedChunk]) -> str:
    context = build_context_block(results)
    return (
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Produce a concise answer grounded only in this context."
    )
