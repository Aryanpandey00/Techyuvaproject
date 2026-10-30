"""
FastAPI Application Entry Point for AI Memory Compression System.
"""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.memories import router as memories_router
from backend.config import get_settings
from core.exceptions import (
    EmbeddingError,
    LLMProviderError,
    MemoryNotFoundError,
    MemorySystemException,
    StorageError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s : %(message)s",
)
logger = logging.getLogger("ai_memory_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown routines."""
    settings = get_settings()
    logger.info("Starting AI Memory Compression API Backend (Env: %s)", settings.APP_ENV)
    logger.info("Vector DB Directory: %s", settings.CHROMA_PERSIST_DIRECTORY)
    logger.info("SQLite DB Path: %s", settings.SQLITE_DB_PATH)
    logger.info("Embedding Model: %s | LLM Provider: %s", settings.EMBEDDING_MODEL_NAME, settings.LLM_PROVIDER)
    yield
    logger.info("Shutting down AI Memory Compression API Backend.")


def create_app() -> FastAPI:
    """Application factory for FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="A Resource-Efficient Long-Term Memory Layer for Generative AI with Intelligent Semantic Retrieval and Context Optimization.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # CORS Middleware configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(MemoryNotFoundError)
    async def memory_not_found_handler(request: Request, exc: MemoryNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "MemoryNotFound", "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "StorageError", "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, exc: EmbeddingError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "EmbeddingError", "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "LLMProviderError", "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(MemorySystemException)
    async def general_memory_exception_handler(request: Request, exc: MemorySystemException):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "MemorySystemError", "message": exc.message, "details": exc.details},
        )

    # Register API Routers
    app.include_router(health_router, prefix=settings.API_V1_PREFIX)
    app.include_router(memories_router, prefix=settings.API_V1_PREFIX)
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", tags=["Root"])
    async def root_index():
        return {
            "application": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "documentation": "/docs",
            "api_v1": settings.API_V1_PREFIX,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
