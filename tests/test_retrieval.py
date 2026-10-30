"""
Unit tests for the Memory Retrieval Pipeline:
- Relevant retrieval
- Irrelevant retrieval
- Top-K behavior
- Multi-factor re-ranking formula: final_score = 0.50 * sim + 0.20 * imp + 0.15 * rec + 0.15 * acc
- Similarity threshold filtering
- Access count and last_accessed_at automatic updates
"""
from datetime import datetime, timezone, timedelta
import pytest
from core.schemas import (
    MemoryCreate,
    MemoryRetrieveRequest,
    MemoryStatus,
    MemoryType,
)
from memory.manager import MemoryManager
from memory.retriever import MemoryRetriever
from memory.scorer import MemoryScorer


def test_relevant_retrieval(memory_manager: MemoryManager):
    """Test retrieving relevant memories matching a semantic query."""
    m_target = memory_manager.create_memory(MemoryCreate(
        content="The user loves building Python microservices with FastAPI.",
        user_id="dev_user",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.9,
    ))
    m_other = memory_manager.create_memory(MemoryCreate(
        content="The user goes hiking every Sunday in the mountains.",
        user_id="dev_user",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.6,
    ))

    resp = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="What Python backend framework does the user like?",
        user_id="dev_user",
        top_k=5,
        similarity_threshold=0.0,
    ))

    assert resp.total_found > 0
    assert len(resp.retrieved_memories) > 0
    top_result = resp.retrieved_memories[0]
    assert top_result.memory.id == m_target.id
    assert top_result.final_score > 0.0
    assert len(resp.context_string) > 0
    assert "FastAPI" in resp.context_string


def test_irrelevant_retrieval_threshold_filtering(memory_manager: MemoryManager):
    """Test that memories below the similarity threshold are rejected."""
    memory_manager.create_memory(MemoryCreate(
        content="User has a pet dog named Max.",
        user_id="dog_owner",
        importance_score=0.5,
    ))

    # Search with very high similarity threshold (0.999) that mock embeddings won't meet
    resp = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="Quantum physics string theory calculations",
        user_id="dog_owner",
        similarity_threshold=0.999,
    ))

    assert resp.total_found == 0
    assert len(resp.retrieved_memories) == 0
    assert resp.context_string == ""


def test_top_k_behavior(memory_manager: MemoryManager):
    """Test that retrieval strictly respects top_k limits."""
    # Seed 10 memories
    for i in range(10):
        memory_manager.create_memory(MemoryCreate(
            content=f"User knowledge fact number {i} about artificial intelligence.",
            user_id="user_topk",
            importance_score=0.5,
        ))

    # Retrieve top 3
    resp_3 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="Tell me about user artificial intelligence facts",
        user_id="user_topk",
        top_k=3,
        similarity_threshold=0.0,
    ))

    assert len(resp_3.retrieved_memories) == 3
    assert resp_3.total_found == 3

    # Retrieve top 7
    resp_7 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="Tell me about user artificial intelligence facts",
        user_id="user_topk",
        top_k=7,
        similarity_threshold=0.0,
    ))

    assert len(resp_7.retrieved_memories) == 7
    assert resp_7.total_found == 7


def test_re_ranking_scoring_formula(memory_manager: MemoryManager):
    """
    Test the multi-factor scoring formula:
    final_score = 0.50 * similarity + 0.20 * importance + 0.15 * recency + 0.15 * access_frequency
    """
    # Create low-importance memory
    low_imp = memory_manager.create_memory(MemoryCreate(
        content="User likes orange soda.",
        user_id="scorer_user",
        importance_score=0.2,
    ))

    # Create high-importance memory
    high_imp = memory_manager.create_memory(MemoryCreate(
        content="User has a severe allergy to tree nuts.",
        user_id="scorer_user",
        importance_score=1.0,
    ))

    scorer = MemoryScorer(
        similarity_weight=0.50,
        importance_weight=0.20,
        recency_weight=0.15,
        access_frequency_weight=0.15,
    )

    now = datetime.now(timezone.utc)
    score_low, _, imp_low, _, _ = scorer.calculate_final_score(0.8, low_imp, now)
    score_high, _, imp_high, _, _ = scorer.calculate_final_score(0.8, high_imp, now)

    assert imp_high > imp_low
    assert score_high > score_low


def test_access_metadata_update_on_retrieval(memory_manager: MemoryManager):
    """Test that last_accessed_at and access_count are updated in SQLite when memories are retrieved."""
    created = memory_manager.create_memory(MemoryCreate(
        content="User prefers Python for data engineering.",
        user_id="u_audit",
        importance_score=0.8,
    ))

    assert created.access_count == 0
    assert created.last_accessed_at is None

    # Execute retrieval
    resp = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="What language does user prefer for data engineering?",
        user_id="u_audit",
        top_k=1,
        similarity_threshold=0.0,
    ))

    assert len(resp.retrieved_memories) == 1
    retrieved_item = resp.retrieved_memories[0]

    # Verify updated in response
    assert retrieved_item.memory.access_count == 1
    assert retrieved_item.memory.last_accessed_at is not None

    # Verify updated in database
    db_item = memory_manager.get_memory(created.id)
    assert db_item.access_count == 1
    assert db_item.last_accessed_at is not None


def test_configurable_weights_retriever(memory_manager: MemoryManager):
    """Test initializing retriever with custom weights."""
    custom_retriever = MemoryRetriever(
        vector_store=memory_manager.chroma_store,
        metadata_store=memory_manager.sqlite_store,
        encoder=memory_manager.encoder,
        similarity_weight=0.70,
        importance_weight=0.10,
        recency_weight=0.10,
        access_frequency_weight=0.10,
    )

    assert custom_retriever.scorer.similarity_weight == 0.70
    assert custom_retriever.scorer.importance_weight == 0.10
