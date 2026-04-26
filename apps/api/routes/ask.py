from fastapi import APIRouter, Request

from apps.api.schemas import AskRequest, AskResponse

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    result = request.app.state.pipeline.ask(
        question=payload.question,
        retriever=payload.retriever,
        top_k=payload.top_k,
        generator=payload.generator,
    )
    return AskResponse(**result)
