"""
Application configuration management using Pydantic Settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration loaded from environment and .env file."""

    # Application
    APP_NAME: str = "A Resource-Efficient Long-Term Memory Layer for Generative AI"
    APP_SHORT_NAME: str = "MemoryLayer AI"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = Field(default="development", alias="APP_ENV")
    DEBUG: bool = Field(default=True, alias="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    # API Server
    API_HOST: str = Field(default="0.0.0.0", alias="API_HOST")
    API_PORT: int = Field(default=8000, alias="API_PORT")
    API_V1_PREFIX: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    # Storage Paths
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = Field(default="llm_conversational_memories", alias="CHROMA_COLLECTION_NAME")
    SQLITE_DB_PATH: str = Field(default="./data/memory_metadata.db", alias="SQLITE_DB_PATH")

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
    EMBEDDING_DEVICE: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # LLM Provider
    LLM_PROVIDER: str = Field(default="mock", alias="LLM_PROVIDER")
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_NAME")
    OPENAI_BASE_URL: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    LOCAL_MODEL_NAME: str = Field(default="meta-llama/Llama-3-8B-Instruct", alias="LOCAL_MODEL_NAME")

    # Memory Hyperparameters
    MEMORY_SIMILARITY_THRESHOLD: float = Field(default=0.75, alias="MEMORY_SIMILARITY_THRESHOLD")
    COMPRESSION_CLUSTER_MIN_SIZE: int = Field(default=2, alias="COMPRESSION_CLUSTER_MIN_SIZE")
    MAX_RETRIEVED_MEMORIES: int = Field(default=5, alias="MAX_RETRIEVED_MEMORIES")
    IMPORTANCE_DECAY_RATE: float = Field(default=0.05, alias="IMPORTANCE_DECAY_RATE")
    FORGETTING_THRESHOLD: float = Field(default=0.20, alias="FORGETTING_THRESHOLD")

    # Resource-Efficient Context Layer & Relevance Gating
    RELEVANCE_THRESHOLD: float = Field(default=0.65, alias="RELEVANCE_THRESHOLD")
    MAX_INJECTED_MEMORIES: int = Field(default=3, alias="MAX_INJECTED_MEMORIES")
    MAX_CONTEXT_CHARACTERS: int = Field(default=1000, alias="MAX_CONTEXT_CHARACTERS")
    MAX_CONTEXT_TOKENS: int = Field(default=250, alias="MAX_CONTEXT_TOKENS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()
