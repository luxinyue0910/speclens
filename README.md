# SpecLens

SpecLens is a local-first, evaluation-driven RAG system for synthetic engineering documents.

Unlike a typical "chat with PDF" demo, SpecLens focuses on retrieval system engineering. It routes each query to keyword search, vector search, or hybrid retrieval, then evaluates retrieval quality, latency, and citation grounding on a synthetic engineering knowledge base.

## Why this project

Pure vector-based RAG is often unstable for exact lookup questions such as API fields, error codes, and configuration values. SpecLens combines:

- Rule-based query routing
- BM25 keyword retrieval
- Qdrant vector retrieval with local embeddings
- Hybrid retrieval with Reciprocal Rank Fusion
- Citation-grounded answer generation
- An experiment runner for RAG evaluation

## MVP features

- Local-first ingestion for markdown engineering docs
- Synthetic Northstar Commerce dataset across PRDs, specs, APIs, incidents, and release notes
- Rule-based router for keyword, vector, or hybrid retrieval
- BM25 keyword retrieval with `rank-bm25`
- Vector retrieval with Qdrant and `all-MiniLM-L6-v2`
- Hybrid retrieval with RRF
- Ollama-backed generation with citation output
- Fallback extractive answer mode when Ollama is unavailable
- Evaluation metrics: route accuracy, retrieval recall, expected doc hit rate, latency, and cost

## Architecture

```text
User Query
  -> Query Router
  -> Keyword / Vector / Hybrid Retrieval
  -> RRF Merge
  -> Answer Generator
  -> Citation Builder
  -> Eval Pipeline
  -> Experiment Report
```

## Project layout

```text
speclens/
  apps/api
  rag/ingestion
  rag/routing
  rag/retrieval
  rag/generation
  rag/evaluation
  data/docs
  data/eval
  reports/experiments
  docs
```

## Quick start

1. Clone the repository and enter it.
2. Copy the environment file.
3. Start Qdrant.
4. Install Python dependencies.
5. Start or verify Ollama locally.
6. Build indexes and run the API.

```bash
cp .env.example .env
make up
make install
ollama serve
make pull-model
make ingest
make dev
```

Ask a question:

```bash
make ask
```

Run evaluation:

```bash
make eval
```

## API

### `POST /ingest`

Builds chunk artifacts from `data/docs` and refreshes the Qdrant collection.

### `POST /ask`

Example request:

```json
{
  "question": "Does checkout support automatic refunds?",
  "retriever": "auto",
  "top_k": 6,
  "generator": "auto"
}
```

### `POST /eval/run`

Runs the experiment pipeline on `data/eval/questions.jsonl` and writes JSON and Markdown reports to `reports/experiments`.

## Example output

Question:

`Does checkout support automatic refunds?`

Expected answer shape:

- Explicitly call out the conflict between the PRD and implementation documents
- Cite the PRD, payment service design, and release notes
- Return retrieved chunks and total latency

## Local-first design

SpecLens can run locally on consumer hardware:

- Local BM25 retrieval
- Local embeddings
- Local Qdrant
- Local Ollama generation
- Local rule-based evaluation

Cloud models can be added later for improved generation quality or judge-based evaluation.

## Roadmap

- Optional cross-encoder reranker
- LLM-based routing
- LLM judge for faithfulness and citation accuracy
- Next.js dashboard
- GitHub Actions regression evals
