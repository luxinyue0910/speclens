from fastapi import APIRouter, Request

from apps.api.schemas import EvalRunRequest, EvalRunResponse
from rag.evaluation.run_experiment import ExperimentConfig, run_experiment

router = APIRouter(tags=["eval"])


@router.post("/eval/run", response_model=EvalRunResponse)
def run_eval(payload: EvalRunRequest, request: Request) -> EvalRunResponse:
    settings = request.app.state.settings
    config = ExperimentConfig(
        retriever=payload.retriever,
        generator=payload.generator,
        top_k=payload.top_k,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    result = run_experiment(config=config, settings=settings)
    return EvalRunResponse(**result)
