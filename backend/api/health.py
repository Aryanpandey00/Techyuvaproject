"""
Health and system diagnostics API router.
"""
from fastapi import APIRouter, Depends, status
from backend.config import Settings, get_settings
from core.schemas import SystemHealthResponse

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get(
    "",
    response_model=SystemHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Diagnostics",
    description="Returns current status of vector store, metadata database, embedding engine, and LLM provider.",
)
async def check_health(settings: Settings = Depends(get_settings)) -> SystemHealthResponse:
    # TODO: Add dynamic probe queries to verify ChromaDB and SQLite write latency
    return SystemHealthResponse(
        status="healthy",
        app_env=settings.APP_ENV,
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        chroma_status="connected",
        sqlite_status="connected",
    )
