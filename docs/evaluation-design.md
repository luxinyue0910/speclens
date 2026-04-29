# Evaluation Design

Phase 1 uses rule-based metrics so the project remains local-first and deterministic.

## Metrics

- `route_accuracy`: predicted route equals the labeled route
- `retrieval_recall`: fraction of expected docs found in retrieved docs
- `expected_doc_hit_rate`: whether at least one expected doc was retrieved
- `image_retrieval_recall`: fraction of expected image sources found in retrieved chunks
- `image_hit_rate`: whether at least one expected image source was retrieved
- `image_citation_hit_rate`: whether at least one expected image source was cited in the final answer
- `cross_modal_coverage`: whether both expected text and image evidence were retrieved for cross-modal questions
- `avg_latency_ms`: average end-to-end latency
- `cost_per_query_usd`: zero for local execution

## Evaluation loop

Each evaluation question provides:

- a question id
- a natural language question
- a query type label
- an expected retrieval route
- a ground-truth answer summary
- expected supporting documents
- optional expected evidence sources

## Multimodal expected source format

Questions can include an `expected_sources` array in addition to `expected_docs`.

Each source expectation is a small object such as:

```json
{
  "source_type": "image",
  "asset_path": "data/assets/diagrams/notification-architecture.svg",
  "doc": "notification-system-design.md"
}
```

or:

```json
{
  "source_type": "text",
  "doc": "notification-system-design.md"
}
```

The experiment runner matches retrieved chunks and citations against these expectations to score image-aware retrieval and grounding.

The experiment runner can compare static retrieval modes against router-based retrieval while keeping the rest of the pipeline fixed.
