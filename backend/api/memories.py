"""
Memory Management REST API endpoints (CRUD, Semantic Search, Status Management, Stats).
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.config import Settings, get_settings
from core.exceptions import MemoryNotFoundError
from core.schemas import (
    CompressionRequest,
    CompressionResult,
    Memory,
    MemoryCreate,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    MemorySearchResult,
    MemoryStatsResponse,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    PruneRequest,
    PruneResult,
)
from memory.manager import MemoryManager

router = APIRouter(prefix="/memories", tags=["Memories"])

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(settings: Settings = Depends(get_settings)) -> MemoryManager:
    """Dependency provider for MemoryManager singleton instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            sqlite_db_path=settings.SQLITE_DB_PATH,
            chroma_dir=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_provider_type=settings.LLM_PROVIDER,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_model_name=settings.OPENAI_MODEL_NAME,
            openai_base_url=settings.OPENAI_BASE_URL,
        )
    return _memory_manager


@router.post(
    "",
    response_model=Memory,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new memory",
)
async def create_memory(
    payload: MemoryCreate,
    manager: MemoryManager = Depends(get_memory_manager),
) -> Memory:
    """Create a new memory item, generate its embedding, and store it in SQLite and ChromaDB."""
    return manager.create_memory(payload)


@router.get(
    "",
    response_model=List[Memory],
    status_code=status.HTTP_200_OK,
    summary="List memories with optional filtering",
)
async def list_memories(
    user_id: Optional[str] = Query(default=None),
    status_filter: Optional[MemoryStatus] = Query(default=None, alias="status"),
    memory_type: Optional[MemoryType] = Query(default=None, alias="type"),
    min_importance: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    search_query: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    manager: MemoryManager = Depends(get_memory_manager),
) -> List[Memory]:
    """Retrieve all stored memories matching filter criteria."""
    return manager.list_memories(
        user_id=user_id,
        status=status_filter,
        memory_type=memory_type,
        min_importance=min_importance,
        min_confidence=min_confidence,
        search_query=search_query,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/stats",
    response_model=MemoryStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get memory database statistics",
)
async def get_stats(
    user_id: Optional[str] = Query(default=None),
    manager: MemoryManager = Depends(get_memory_manager),
) -> MemoryStatsResponse:
    """Return aggregate statistics of memories."""
    return manager.get_stats(user_id=user_id)


@router.post(
    "/search",
    response_model=List[MemorySearchResult],
    status_code=status.HTTP_200_OK,
    summary="Semantic similarity search",
)
async def search_memories(
    payload: MemoryRetrieveRequest,
    manager: MemoryManager = Depends(get_memory_manager),
) -> List[MemorySearchResult]:
    """Perform semantic vector similarity search."""
    return manager.search_similar(
        query=payload.query,
        user_id=payload.user_id,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity,
        memory_type=payload.memory_type,
        status=payload.status,
    )


@router.post(
    "/retrieve",
    response_model=MemoryRetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve memories with formatted prompt context",
)
async def retrieve_memories(
    payload: MemoryRetrieveRequest,
    manager: MemoryManager = Depends(get_memory_manager),
) -> MemoryRetrieveResponse:
    """Retrieve relevant memories and format as context block."""
    return manager.retrieve_context(payload)


@router.post(
    "/compress",
    response_model=CompressionResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger memory clustering and compression",
)
async def compress_memory_items(
    payload: CompressionRequest,
    manager: MemoryManager = Depends(get_memory_manager),
) -> CompressionResult:
    """Execute memory compression on clusters or explicit memory IDs."""
    return manager.compress_memories(payload)


@router.post(
    "/prune",
    response_model=PruneResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger memory forgetting/pruning lifecycle",
)
async def prune_memory_items(
    payload: Optional[PruneRequest] = None,
    manager: MemoryManager = Depends(get_memory_manager),
) -> PruneResult:
    """Execute decay-based and inactivity-based pruning of obsolete memories."""
    return manager.prune_memories(payload or PruneRequest())


@router.get(
    "/{memory_id}",
    response_model=Memory,
    status_code=status.HTTP_200_OK,
    summary="Get memory by ID",
)
async def get_memory(
    memory_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
) -> Memory:
    """Fetch details of a single memory by ID."""
    memory = manager.get_memory(memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with ID '{memory_id}' not found."
        )
    return memory


@router.put(
    "/{memory_id}",
    response_model=Memory,
    status_code=status.HTTP_200_OK,
    summary="Update memory item",
)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    manager: MemoryManager = Depends(get_memory_manager),
) -> Memory:
    """Update memory content, type, importance, confidence, or status."""
    try:
        return manager.update_memory(memory_id, payload)
    except MemoryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        ) from e


@router.patch(
    "/{memory_id}/status",
    response_model=Memory,
    status_code=status.HTTP_200_OK,
    summary="Update memory status",
)
async def update_memory_status(
    memory_id: str,
    status_val: MemoryStatus = Query(...),
    manager: MemoryManager = Depends(get_memory_manager),
) -> Memory:
    """Update memory lifecycle status."""
    try:
        return manager.update_status(memory_id, status_val)
    except MemoryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        ) from e


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete memory item",
)
async def delete_memory(
    memory_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
) -> None:
    """Permanently delete a memory item and its vector embedding."""
    success = manager.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with ID '{memory_id}' not found."
        )
