"""
Pytest configuration and shared test fixtures for memory storage tests.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from database.memory_storage import MemoryStorageService
from embeddings.encoder import MockEmbeddingEncoder
from llm.provider import MockLLMProvider
from memory.manager import MemoryManager


@pytest.fixture
def mock_encoder():
    """Deterministic 384-dimensional mock embedding encoder."""
    return MockEmbeddingEncoder(dim=384)


@pytest.fixture
def mock_llm():
    """Pattern-based mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def memory_storage(mock_encoder):
    """Isolated in-memory MemoryStorageService for testing."""
    return MemoryStorageService(
        sqlite_db_path=":memory:",
        chroma_persist_dir=None,
        encoder=mock_encoder,
        in_memory=True,
    )


@pytest.fixture
def memory_manager():
    """In-memory isolated MemoryManager for testing."""
    return MemoryManager(
        sqlite_db_path=":memory:",
        chroma_dir=None,
        embedding_model="mock",
        llm_provider_type="mock",
        in_memory_db=True,
    )


@pytest.fixture
def test_client():
    """FastAPI TestClient for making API requests."""
    return TestClient(app)
