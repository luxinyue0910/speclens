# Evaluation Design

Phase 1 uses rule-based metrics so the project remains local-first and deterministic.

## Metrics

- `route_accuracy`: predicted route equals the labeled route
- `retrieval_recall`: fraction of expected docs found in retrieved docs
- `expected_doc_hit_rate`: whether at least one expected doc was retrieved
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

The experiment runner can compare static retrieval modes against router-based retrieval while keeping the rest of the pipeline fixed.
