from fastapi import FastAPI

from apps.api.routes import ask, eval, ingest
from rag.config import ensure_runtime_directories, get_settings
from rag.pipeline import RagPipeline

settings = get_settings()
ensure_runtime_directories(settings)

app = FastAPI(
    title="SpecLens API",
    version="0.1.0",
    description="Local-first hybrid retrieval and evaluation for engineering docs.",
)
app.state.settings = settings
app.state.pipeline = RagPipeline(settings=settings)

app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(eval.router)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
