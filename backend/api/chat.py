"""
Chat completion API endpoint integrated with conversational memory retrieval and extraction.
"""
from fastapi import APIRouter, Depends, status
from backend.api.memories import get_memory_manager
from core.schemas import ChatRequest, ChatResponse
from memory.manager import MemoryManager

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with long-term memory",
    description="Processes a user prompt by retrieving relevant semantic memories, generating an augmented LLM response, and extracting new memory items.",
)
async def chat_completion(
    payload: ChatRequest,
    manager: MemoryManager = Depends(get_memory_manager),
) -> ChatResponse:
    """Execute end-to-end memory-augmented conversational turn."""
    return manager.process_chat(payload)
