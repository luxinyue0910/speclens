PYTHON ?= python3

.PHONY: install up down pull-model ingest ask eval dev

install:
	$(PYTHON) -m pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

pull-model:
	ollama pull phi3

ingest:
	$(PYTHON) -m rag.ingestion.build_indexes

ask:
	curl -X POST http://localhost:8000/ask \
	  -H "Content-Type: application/json" \
	  -d '{"question":"Does checkout support automatic refunds?","retriever":"auto","top_k":6,"generator":"auto"}'

eval:
	$(PYTHON) -m rag.evaluation.run_experiment --retriever auto --generator auto --top-k 6

dev:
	uvicorn apps.api.main:app --reload
