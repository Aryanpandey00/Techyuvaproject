"""
Unit tests for the Memory Storage Layer:
- Creating memories
- Embedding memories
- Storing memories (SQLite and ChromaDB)
- Semantic similarity search
- Updating memories
- Deleting memories
- Memory types support and status management
"""
import pytest
from core.schemas import MemoryCreate, MemoryStatus, MemoryType, MemoryUpdate
from database.memory_storage import MemoryStorageService
from embeddings.encoder import MockEmbeddingEncoder


def test_create_memory(memory_storage: MemoryStorageService):
    """Test creating a memory with all required fields."""
    req = MemoryCreate(
        content="I am developing an AI memory system using Python and FastAPI.",
        user_id="user_123",
        memory_type=MemoryType.PROJECT,
        importance_score=0.9,
        confidence_score=0.95,
        status=MemoryStatus.ACTIVE,
    )
    memory = memory_storage.create_memory(req)

    assert memory.id is not None
    assert memory.user_id == "user_123"
    assert memory.content == req.content
    assert memory.memory_type == MemoryType.PROJECT
    assert memory.importance_score == 0.9
    assert memory.confidence_score == 0.95
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.access_count == 0
    assert memory.created_at is not None
    assert memory.updated_at is not None
    assert memory.last_accessed_at is None


def test_embedding_memory(mock_encoder: MockEmbeddingEncoder):
    """Test embedding single text and batch texts."""
    text = "The user prefers dark mode for IDE interfaces."
    vec = mock_encoder.encode_text(text)

    assert isinstance(vec, list)
    assert len(vec) == mock_encoder.dimension
    assert len(vec) == 384

    # Test batch embedding
    texts = [
        "Goal: complete final year project by April.",
        "Fact: User resides in Seattle.",
    ]
    batch_vecs = mock_encoder.encode_batch(texts)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384


def test_store_memory_in_sqlite_and_chroma(memory_storage: MemoryStorageService):
    """Test that a created memory is persisted in both SQLite and ChromaDB."""
    req = MemoryCreate(
        content="User goal is to achieve 90% retrieval accuracy.",
        user_id="alice",
        memory_type=MemoryType.GOAL,
        importance_score=0.85,
        confidence_score=1.0,
    )
    created = memory_storage.create_memory(req)

    # Verify SQLite persistence
    sqlite_memory = memory_storage.sqlite_store.get_memory(created.id)
    assert sqlite_memory is not None
    assert sqlite_memory.id == created.id
    assert sqlite_memory.content == req.content
    assert sqlite_memory.user_id == "alice"
    assert sqlite_memory.memory_type == MemoryType.GOAL

    # Verify ChromaDB persistence
    chroma_item = memory_storage.chroma_store.get_by_id(created.id)
    assert chroma_item is not None
    assert chroma_item["id"] == created.id
    assert chroma_item["document"] == req.content
    assert chroma_item["metadata"]["user_id"] == "alice"
    assert chroma_item["metadata"]["memory_type"] == "goal"


def test_semantic_similarity_search(memory_storage: MemoryStorageService):
    """Test semantic similarity search across stored memories."""
    # Seed memories
    m1 = memory_storage.create_memory(MemoryCreate(
        content="I love developing backend microservices with FastAPI in Python.",
        user_id="user_dev",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.9,
    ))
    m2 = memory_storage.create_memory(MemoryCreate(
        content="My favorite outdoor activity is hiking in the mountains.",
        user_id="user_dev",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.6,
    ))

    # Search for backend programming query
    results = memory_storage.search_similar_memories(
        query="What Python backend framework does the user like?",
        user_id="user_dev",
        top_k=2,
        min_similarity=0.0,
    )

    assert len(results) > 0
    assert any(r.memory.id == m1.id for r in results)
    assert results[0].similarity_score >= 0.0
    assert results[0].similarity_score <= 1.0

    # Verify access was recorded in SQLite
    refetched = memory_storage.get_memory(results[0].memory.id)
    assert refetched.access_count >= 1
    assert refetched.last_accessed_at is not None


def test_update_memory(memory_storage: MemoryStorageService):
    """Test updating content, importance, confidence, and status."""
    created = memory_storage.create_memory(MemoryCreate(
        content="Initial temporary note about meeting.",
        user_id="bob",
        memory_type=MemoryType.TEMPORARY,
        importance_score=0.3,
        confidence_score=0.5,
    ))

    update_req = MemoryUpdate(
        content="Updated meeting note: rescheduled to 3 PM Friday.",
        importance_score=0.8,
        confidence_score=0.95,
        status=MemoryStatus.ARCHIVED,
    )
    updated = memory_storage.update_memory(created.id, update_req)

    assert updated.content == "Updated meeting note: rescheduled to 3 PM Friday."
    assert updated.importance_score == 0.8
    assert updated.confidence_score == 0.95
    assert updated.status == MemoryStatus.ARCHIVED

    # Verify updated in SQLite
    sqlite_check = memory_storage.get_memory(created.id)
    assert sqlite_check.content == updated.content
    assert sqlite_check.status == MemoryStatus.ARCHIVED

    # Verify updated in ChromaDB
    chroma_check = memory_storage.chroma_store.get_by_id(created.id)
    assert chroma_check["document"] == updated.content
    assert chroma_check["metadata"]["status"] == "archived"


def test_delete_memory(memory_storage: MemoryStorageService):
    """Test deleting memory from both SQLite and ChromaDB."""
    created = memory_storage.create_memory(MemoryCreate(
        content="Obsolete memory that should be deleted.",
        user_id="charlie",
    ))

    assert memory_storage.get_memory(created.id) is not None
    assert memory_storage.chroma_store.get_by_id(created.id) is not None

    deleted = memory_storage.delete_memory(created.id)
    assert deleted is True

    assert memory_storage.get_memory(created.id) is None
    assert memory_storage.chroma_store.get_by_id(created.id) is None


def test_supported_memory_types(memory_storage: MemoryStorageService):
    """Verify that all 7 required memory types are supported."""
    types = [
        MemoryType.FACT,
        MemoryType.PREFERENCE,
        MemoryType.GOAL,
        MemoryType.PROJECT,
        MemoryType.PERSONAL,
        MemoryType.TEMPORARY,
        MemoryType.CONVERSATION_SUMMARY,
    ]

    for t in types:
        mem = memory_storage.create_memory(MemoryCreate(
            content=f"Test memory of type {t.value}",
            user_id="test_user",
            memory_type=t,
        ))
        assert mem.memory_type == t
        fetched = memory_storage.get_memory(mem.id)
        assert fetched.memory_type == t


def test_list_memories_filtering(memory_storage: MemoryStorageService):
    """Test listing memories with multiple query and status filters."""
    memory_storage.create_memory(MemoryCreate(
        content="Fact 1 about python",
        user_id="user_a",
        memory_type=MemoryType.FACT,
        importance_score=0.9,
    ))
    memory_storage.create_memory(MemoryCreate(
        content="Goal 1 about python",
        user_id="user_a",
        memory_type=MemoryType.GOAL,
        importance_score=0.4,
    ))
    memory_storage.create_memory(MemoryCreate(
        content="Fact 2 for user b",
        user_id="user_b",
        memory_type=MemoryType.FACT,
        importance_score=0.8,
    ))

    # Filter by user_id
    user_a_items = memory_storage.list_memories(user_id="user_a")
    assert len(user_a_items) == 2

    # Filter by user_id and min_importance
    high_imp = memory_storage.list_memories(user_id="user_a", min_importance=0.8)
    assert len(high_imp) == 1
    assert high_imp[0].importance_score == 0.9

    # Filter by memory_type
    goal_items = memory_storage.list_memories(user_id="user_a", memory_type=MemoryType.GOAL)
    assert len(goal_items) == 1
    assert goal_items[0].memory_type == MemoryType.GOAL
