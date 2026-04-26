from fastapi import APIRouter, Request

from apps.api.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(payload: IngestRequest, request: Request) -> IngestResponse:
    result = request.app.state.pipeline.ingest(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return IngestResponse(**result)
