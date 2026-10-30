"""
API router package initialization.
"""
from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.memories import router as memories_router

__all__ = ["chat_router", "health_router", "memories_router"]
