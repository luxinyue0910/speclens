from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field


RetrieverMode = Literal["auto", "keyword", "vector", "hybrid"]
GeneratorMode = Literal["auto", "ollama", "extractive"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    retriever: RetrieverMode = "auto"
    top_k: int = Field(default=6, ge=1, le=20)
    generator: GeneratorMode = "auto"


class CitationResponse(BaseModel):
    doc: str
    chunk_id: str
    claim: str
    source_type: Optional[str] = None
    asset_path: Optional[str] = None


class RetrievedChunkResponse(BaseModel):
    doc: str
    chunk_id: str
    score: float
    text: str
    source_type: Optional[str] = None
    asset_path: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    route: str
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]
    latency_ms: int
    confidence: str


class IngestRequest(BaseModel):
    chunk_size: int = Field(default=220, ge=50, le=2000)
    chunk_overlap: int = Field(default=40, ge=0, le=400)


class IngestResponse(BaseModel):
    document_count: int
    chunk_count: int
    collection_name: str
    manifest_path: str


class EvalRunRequest(BaseModel):
    retriever: RetrieverMode = "auto"
    generator: GeneratorMode = "auto"
    top_k: int = Field(default=6, ge=1, le=20)
    chunk_size: int = Field(default=220, ge=50, le=2000)
    chunk_overlap: int = Field(default=40, ge=0, le=400)


class EvalRunResponse(BaseModel):
    experiment_name: str
    questions: int
    route_accuracy: float
    retrieval_recall: float
    expected_doc_hit_rate: float
    multimodal_question_count: int
    cross_modal_question_count: int
    image_retrieval_recall: float
    image_hit_rate: float
    image_citation_hit_rate: float
    cross_modal_coverage: float
    avg_latency_ms: float
    cost_per_query_usd: float
    report_json: str
    report_markdown: str
