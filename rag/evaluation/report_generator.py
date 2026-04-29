from __future__ import annotations

import json
from pathlib import Path


def write_report_files(report: dict[str, object], reports_dir: Path) -> tuple[str, str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = str(report["experiment_name"])
    json_path = reports_dir / f"{experiment_name}.json"
    md_path = reports_dir / f"{experiment_name}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# {experiment_name}",
        "",
        f"- Questions: {report['questions']}",
        f"- Route Accuracy: {report['route_accuracy']:.2f}",
        f"- Retrieval Recall: {report['retrieval_recall']:.2f}",
        f"- Expected Doc Hit Rate: {report['expected_doc_hit_rate']:.2f}",
        f"- Multimodal Questions: {report['multimodal_question_count']}",
        f"- Image Retrieval Recall: {report['image_retrieval_recall']:.2f}",
        f"- Image Hit Rate: {report['image_hit_rate']:.2f}",
        f"- Image Citation Hit Rate: {report['image_citation_hit_rate']:.2f}",
        f"- Cross-Modal Coverage: {report['cross_modal_coverage']:.2f}",
        f"- Avg Latency (ms): {report['avg_latency_ms']:.2f}",
        f"- Cost / Query (USD): {report['cost_per_query_usd']:.3f}",
        "",
        "## Per-question summary",
        "",
    ]
    for row in report["results"]:
        md_lines.append(
            f"- `{row['id']}` route={row['predicted_route']} recall={row['retrieval_recall']:.2f} "
            f"hit={int(row['expected_doc_hit'])} latency_ms={row['latency_ms']}"
        )
        if row.get("has_image_expectation"):
            md_lines.append(
                f"  image_recall={row['image_retrieval_recall']:.2f} "
                f"image_hit={int(row['image_hit'])} "
                f"image_citation_hit={int(row['image_citation_hit'])}"
            )
        if row.get("has_cross_modal_expectation"):
            md_lines.append(
                f"  cross_modal_coverage={row['cross_modal_coverage']:.2f}"
            )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)
