from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.evaluation.metrics import (
    compute_expected_doc_hit_rate,
    compute_retrieval_recall,
)
from rag.evaluation.report_generator import write_report_files
from rag.pipeline import RagPipeline


@dataclass(frozen=True)
class ExperimentConfig:
    retriever: str = "auto"
    generator: str = "auto"
    top_k: int = 6
    chunk_size: int = 220
    chunk_overlap: int = 40


def _load_questions(path) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def run_experiment(config: ExperimentConfig, settings: Settings | None = None) -> dict[str, object]:
    resolved_settings = settings or get_settings()
    pipeline = RagPipeline(settings=resolved_settings)
    pipeline.ingest(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    questions = _load_questions(resolved_settings.eval_questions_path)
    rows: list[dict[str, object]] = []

    for item in questions:
        result = pipeline.ask(
            question=str(item["question"]),
            retriever=config.retriever,
            top_k=config.top_k,
            generator=config.generator,
        )
        retrieved_docs = [chunk["doc"] for chunk in result["retrieved_chunks"]]
        recall = compute_retrieval_recall(
            expected_docs=list(item["expected_docs"]),
            retrieved_docs=retrieved_docs,
        )
        hit_rate = compute_expected_doc_hit_rate(
            expected_docs=list(item["expected_docs"]),
            retrieved_docs=retrieved_docs,
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_route": item["expected_route"],
                "predicted_route": result["route"],
                "route_correct": result["route"] == item["expected_route"],
                "retrieval_recall": recall,
                "expected_doc_hit": hit_rate,
                "latency_ms": result["latency_ms"],
            }
        )

    question_count = len(rows)
    route_accuracy = sum(1 for row in rows if row["route_correct"]) / question_count
    retrieval_recall = sum(row["retrieval_recall"] for row in rows) / question_count
    expected_doc_hit_rate = sum(row["expected_doc_hit"] for row in rows) / question_count
    avg_latency_ms = sum(row["latency_ms"] for row in rows) / question_count

    experiment_name = (
        f"{config.retriever}_router_chunk{config.chunk_size}_top{config.top_k}_{config.generator}"
    )
    report = {
        "experiment_name": experiment_name,
        "questions": question_count,
        "route_accuracy": round(route_accuracy, 4),
        "retrieval_recall": round(retrieval_recall, 4),
        "expected_doc_hit_rate": round(expected_doc_hit_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "cost_per_query_usd": 0.0,
        "results": rows,
    }
    report_json, report_markdown = write_report_files(report, resolved_settings.reports_dir)
    report["report_json"] = report_json
    report["report_markdown"] = report_markdown
    return report


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run a SpecLens experiment.")
    parser.add_argument("--retriever", default="auto")
    parser.add_argument("--generator", default="auto")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    args = parser.parse_args()
    return ExperimentConfig(
        retriever=args.retriever,
        generator=args.generator,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def main() -> None:
    config = _parse_args()
    report = run_experiment(config=config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
