# Architecture

SpecLens is structured as a lightweight retrieval pipeline that favors local execution and clear extension points.

## Flow

1. Markdown docs under `data/docs` are loaded and chunked.
2. Chunk metadata is saved locally under `data/indexes`.
3. Chunks are embedded with a sentence-transformer and written to Qdrant.
4. Query routing selects `keyword`, `vector`, or `hybrid`.
5. Retrieval returns scored chunks.
6. The generator produces a citation-grounded answer.
7. The evaluation runner executes the full pipeline on a labeled dataset and writes reports.

## Design choices

- BM25 handles exact term lookup and API-style questions.
- Vector retrieval covers semantic and paraphrased questions.
- Hybrid retrieval resolves mixed or conflicting evidence better than either single strategy alone.
- The API is thin by design. Core orchestration lives in `rag/pipeline.py` so the same logic can power both HTTP routes and CLI experiments.
