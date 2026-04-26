# Search Indexing Architecture

## Summary

The search platform builds both lexical and semantic representations for product search.

## Pipeline

1. Product updates are written to the catalog topic.
2. The indexing worker normalizes attributes and category labels.
3. BM25-compatible fields are written to the lexical index.
4. Product text is embedded and stored for semantic ranking features.

## Operational decisions

- Search Ranking v2 uses a larger lexical candidate set before model scoring.
- Embedding refreshes run asynchronously and may lag behind catalog writes by a few minutes.
- Candidate generation prefers recall over latency because the ranking model filters aggressively later.

## Configuration

- `SEARCH_INDEX_BATCH_SIZE` defaults to `200`
- `SEARCH_FEATURE_CACHE_TTL_MS` defaults to `15000`
- `SEARCH_EMBEDDING_REFRESH_INTERVAL_MINUTES` defaults to `10`
