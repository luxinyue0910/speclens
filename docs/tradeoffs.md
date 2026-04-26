# Tradeoffs

## Why rule-based routing first

Rule-based routing is easier to debug, deterministic, and good enough for an MVP. It also makes route accuracy measurable without introducing model variance.

## Why persist chunk metadata locally

The dataset is intentionally small. Persisting chunk metadata to local JSONL keeps the BM25 path simple and inspectable while Qdrant holds the semantic index.

## Why keep reranking optional

Cross-encoder reranking can improve precision, but it adds model overhead and complexity. The initial version keeps a reranker stub so the project can evolve without forcing extra dependencies into the happy path.

## Why include an extractive fallback

The primary generation path is Ollama, but local environments are often incomplete on day one. An extractive fallback keeps `/ask` and the eval loop usable even before a model is running.
