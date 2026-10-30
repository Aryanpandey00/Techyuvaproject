"""
Unit and integration tests for Memory Retention and Forgetting System:
- Pinned memory protection
- High-importance memory protection
- Decay-based forgetting
- Inactivity-based forgetting
- Soft-delete persistence (SQLite FORGOTTEN status, Chroma eviction)
- Hard-delete removal
- Forgotten memories exclusion from active retrieval and search
- Lifecycle decision tree (KEEP / ARCHIVE / FORGET)
- User-scoped pruning
- API endpoint integration
"""
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from core.schemas import (
    MemoryCreate,
    MemoryRetrieveRequest,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    PruneRequest,
    PruneResult,
)
from memory.forgetter import MemoryForgetter
from memory.manager import MemoryManager
from memory.scorer import MemoryScorer


def test_pinned_memory_protection(memory_manager: MemoryManager):
    """Test that pinned memories are never pruned or forgotten even if decayed or old."""
    now = datetime.now(timezone.utc)
    ninety_days_ago = now - timedelta(days=90)

    # Create pinned memory with low intrinsic importance
    mem = memory_manager.create_memory(MemoryCreate(
        content="Core system instruction: always speak respectfully.",
        user_id="user_pin",
        memory_type=MemoryType.FACT,
        importance_score=0.10,
        metadata={"is_pinned": True},
    ))
    assert mem.is_pinned is True

    # Artificially age the memory in SQLite
    memory_manager.sqlite_store.update_memory(
        mem.id,
        MemoryUpdate(importance_score=0.10)
    )
    # Set created_at and updated_at directly in DB to 90 days ago
    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (ninety_days_ago.isoformat(), ninety_days_ago.isoformat(), ninety_days_ago.isoformat(), mem.id)
        )
        conn.commit()

    reloaded = memory_manager.get_memory(mem.id)
    assert reloaded.is_pinned is True

    # 1. Evaluation should return KEEP and not prune
    decision = memory_manager.forgetter.decide_lifecycle(reloaded, threshold=0.20, max_inactivity_days=30, now=now)
    assert decision == "KEEP"

    should_prune = memory_manager.forgetter.evaluate_memory_for_pruning(reloaded, threshold=0.20, max_inactivity_days=30, now=now)
    assert should_prune is False

    # 2. Run pruning pass
    prune_res = memory_manager.prune_memories(PruneRequest(user_id="user_pin", importance_threshold=0.50))
    assert prune_res.pruned_count == 0
    assert mem.id not in prune_res.pruned_memory_ids

    # Memory must remain ACTIVE
    check = memory_manager.get_memory(mem.id)
    assert check is not None
    assert check.status == MemoryStatus.ACTIVE


def test_high_importance_memory_protection(memory_manager: MemoryManager):
    """Test that very important memories (importance >= 0.90) are strictly protected."""
    now = datetime.now(timezone.utc)
    hundred_days_ago = now - timedelta(days=100)

    mem = memory_manager.create_memory(MemoryCreate(
        content="Critical user identity: User is allergic to penicillin.",
        user_id="user_vip",
        memory_type=MemoryType.FACT,
        importance_score=0.95,
    ))

    # Artificially age the memory
    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (hundred_days_ago.isoformat(), hundred_days_ago.isoformat(), hundred_days_ago.isoformat(), mem.id)
        )
        conn.commit()

    reloaded = memory_manager.get_memory(mem.id)
    assert reloaded.importance_score >= 0.90

    decision = memory_manager.forgetter.decide_lifecycle(reloaded, threshold=0.30, max_inactivity_days=30, now=now)
    assert decision == "KEEP"

    should_prune = memory_manager.forgetter.evaluate_memory_for_pruning(reloaded, threshold=0.30, max_inactivity_days=30, now=now)
    assert should_prune is False

    prune_res = memory_manager.prune_memories(PruneRequest(user_id="user_vip", importance_threshold=0.90))
    assert prune_res.pruned_count == 0

    check = memory_manager.get_memory(mem.id)
    assert check.status == MemoryStatus.ACTIVE


def test_decay_based_forgetting(memory_manager: MemoryManager):
    """Test that low-importance memories with decayed score below threshold are forgotten."""
    now = datetime.now(timezone.utc)
    forty_days_ago = now - timedelta(days=40)

    mem = memory_manager.create_memory(MemoryCreate(
        content="Temporary wifi password for guest network is winter2024.",
        user_id="user_temp",
        memory_type=MemoryType.TEMPORARY,
        importance_score=0.25,
    ))

    # Artificially age the memory by 40 days
    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (forty_days_ago.isoformat(), forty_days_ago.isoformat(), mem.id)
        )
        conn.commit()

    reloaded = memory_manager.get_memory(mem.id)

    # Decayed importance = 0.25 * exp(-0.05 * 40) = 0.25 * exp(-2.0) ≈ 0.25 * 0.1353 ≈ 0.0338 < 0.20
    decayed_score = memory_manager.scorer.calculate_decayed_importance(reloaded, now=now)
    assert decayed_score < 0.20

    decision = memory_manager.forgetter.decide_lifecycle(reloaded, threshold=0.20, now=now)
    assert decision == "FORGET"

    should_prune = memory_manager.forgetter.evaluate_memory_for_pruning(reloaded, threshold=0.20, now=now)
    assert should_prune is True


def test_inactivity_based_forgetting(memory_manager: MemoryManager):
    """Test that inactive low-importance memories exceeding max_inactivity_days are forgotten."""
    now = datetime.now(timezone.utc)
    sixty_days_ago = now - timedelta(days=60)

    # Moderate-low importance memory (0.50 < 0.60 cutoff)
    mem_inactive = memory_manager.create_memory(MemoryCreate(
        content="User mentioned they might read a book about gardening someday.",
        user_id="user_inactive",
        importance_score=0.50,
    ))

    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (sixty_days_ago.isoformat(), sixty_days_ago.isoformat(), sixty_days_ago.isoformat(), mem_inactive.id)
        )
        conn.commit()

    reloaded = memory_manager.get_memory(mem_inactive.id)

    # With threshold 0.01 (so decay alone won't trigger), inactivity of 60 > 30 days triggers FORGET
    decision = memory_manager.forgetter.decide_lifecycle(
        reloaded,
        threshold=0.01,
        max_inactivity_days=30,
        now=now,
    )
    assert decision == "FORGET"

    # In contrast, an inactive memory with higher importance (e.g. 0.75) is NOT forgotten by inactivity alone
    mem_important = memory_manager.create_memory(MemoryCreate(
        content="User graduated from University of Washington in 2020.",
        user_id="user_inactive",
        importance_score=0.75,
    ))
    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (sixty_days_ago.isoformat(), sixty_days_ago.isoformat(), sixty_days_ago.isoformat(), mem_important.id)
        )
        conn.commit()

    reloaded_imp = memory_manager.get_memory(mem_important.id)
    decision_imp = memory_manager.forgetter.decide_lifecycle(
        reloaded_imp,
        threshold=0.01,
        max_inactivity_days=30,
        now=now,
    )
    # Higher importance (> 0.60) is not FORGET, but ARCHIVE
    assert decision_imp != "FORGET"


def test_soft_delete_lifecycle(memory_manager: MemoryManager):
    """Test soft delete: updates SQLite to FORGOTTEN, evicts from Chroma, keeps record inspectable."""
    mem = memory_manager.create_memory(MemoryCreate(
        content="Obsolete note to be soft-deleted.",
        user_id="user_soft",
        importance_score=0.15,
    ))

    # Verify initial active state in SQLite and Chroma
    assert memory_manager.get_memory(mem.id).status == MemoryStatus.ACTIVE
    assert memory_manager.chroma_store.get_by_id(mem.id) is not None

    # Execute soft pruning
    res = memory_manager.prune_memories(PruneRequest(
        user_id="user_soft",
        importance_threshold=0.50,
        hard_delete=False,
    ))

    assert res.pruned_count == 1
    assert mem.id in res.pruned_memory_ids

    # 1. SQLite keeps record with FORGOTTEN status
    saved = memory_manager.get_memory(mem.id)
    assert saved is not None
    assert saved.status == MemoryStatus.FORGOTTEN

    # 2. Chroma vector entry is evicted
    assert memory_manager.chroma_store.get_by_id(mem.id) is None

    # 3. Stats reflect forgotten count
    stats = memory_manager.get_stats(user_id="user_soft")
    assert stats.forgotten_count == 1
    assert stats.active_count == 0


def test_hard_delete_lifecycle(memory_manager: MemoryManager):
    """Test hard delete: permanently deletes record from SQLite and ChromaDB."""
    mem = memory_manager.create_memory(MemoryCreate(
        content="Obsolete note to be hard-deleted permanently.",
        user_id="user_hard",
        importance_score=0.15,
    ))

    assert memory_manager.get_memory(mem.id) is not None
    assert memory_manager.chroma_store.get_by_id(mem.id) is not None

    # Execute hard pruning
    res = memory_manager.prune_memories(PruneRequest(
        user_id="user_hard",
        importance_threshold=0.50,
        hard_delete=True,
    ))

    assert res.pruned_count == 1
    assert mem.id in res.pruned_memory_ids

    # 1. SQLite record is permanently deleted
    assert memory_manager.get_memory(mem.id) is None

    # 2. Chroma entry is deleted
    assert memory_manager.chroma_store.get_by_id(mem.id) is None

    # 3. Stats reflect 0 total memories
    stats = memory_manager.get_stats(user_id="user_hard")
    assert stats.total_memories == 0


def test_forgotten_memories_not_retrieved_as_active(memory_manager: MemoryManager):
    """Test that forgotten memories are never returned in active list, vector search, or context retrieval."""
    m_active = memory_manager.create_memory(MemoryCreate(
        content="Python is the primary language for this project.",
        user_id="user_search",
        importance_score=0.90,
    ))
    m_forgotten = memory_manager.create_memory(MemoryCreate(
        content="Python 2.7 legacy configuration notes.",
        user_id="user_search",
        importance_score=0.10,
    ))

    # Soft prune m_forgotten
    memory_manager.prune_memories(PruneRequest(
        user_id="user_search",
        importance_threshold=0.20,
        hard_delete=False,
    ))

    # Verify status
    assert memory_manager.get_memory(m_active.id).status == MemoryStatus.ACTIVE
    assert memory_manager.get_memory(m_forgotten.id).status == MemoryStatus.FORGOTTEN

    # 1. Listing active memories must exclude forgotten memory
    active_list = memory_manager.list_memories(
        user_id="user_search",
        status=MemoryStatus.ACTIVE,
    )
    active_ids = [m.id for m in active_list]
    assert m_active.id in active_ids
    assert m_forgotten.id not in active_ids

    # 2. Semantic similarity search must exclude forgotten memory
    similar_results = memory_manager.search_similar(
        query="Python configuration language",
        user_id="user_search",
        status=MemoryStatus.ACTIVE,
    )
    result_ids = [r.memory.id for r in similar_results]
    assert m_active.id in result_ids
    assert m_forgotten.id not in result_ids

    # 3. Retriever context generation must exclude forgotten memory
    retrieve_resp = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="Python configuration language",
        user_id="user_search",
    ))
    retrieved_ids = [r.memory.id for r in retrieve_resp.retrieved_memories]
    assert m_active.id in retrieved_ids
    assert m_forgotten.id not in retrieved_ids
    assert "Python 2.7" not in retrieve_resp.context_string


def test_forgetter_lifecycle_decisions(memory_manager: MemoryManager):
    """Verify individual lifecycle classifications: KEEP, ARCHIVE, FORGET."""
    now = datetime.now(timezone.utc)
    ninety_days_ago = now - timedelta(days=90)

    # 1. Pinned -> KEEP
    m_pinned = MemoryCreate(content="Pinned fact", metadata={"is_pinned": True})
    mem_p = memory_manager.create_memory(m_pinned)
    assert memory_manager.forgetter.decide_lifecycle(mem_p, now=now) == "KEEP"

    # 2. High importance (>= 0.90) -> KEEP
    m_high = MemoryCreate(content="VIP fact", importance_score=0.92)
    mem_h = memory_manager.create_memory(m_high)
    assert memory_manager.forgetter.decide_lifecycle(mem_h, now=now) == "KEEP"

    # 3. Decayed below threshold -> FORGET
    m_decay = memory_manager.create_memory(MemoryCreate(content="Decaying fact", importance_score=0.15))
    assert memory_manager.forgetter.decide_lifecycle(m_decay, threshold=0.20, now=now) == "FORGET"

    # 4. Old inactive with moderate importance -> ARCHIVE
    m_arch = memory_manager.create_memory(MemoryCreate(content="Old project fact", importance_score=0.70))
    with memory_manager.sqlite_store._get_connection() as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (ninety_days_ago.isoformat(), ninety_days_ago.isoformat(), ninety_days_ago.isoformat(), m_arch.id)
        )
        conn.commit()
    reloaded_arch = memory_manager.get_memory(m_arch.id)
    decision = memory_manager.forgetter.decide_lifecycle(
        reloaded_arch,
        threshold=0.01,
        max_inactivity_days=30,
        now=now,
    )
    assert decision == "ARCHIVE"


def test_prune_filtered_by_user_id(memory_manager: MemoryManager):
    """Verify that pruning respects user_id filtering."""
    m_u1 = memory_manager.create_memory(MemoryCreate(
        content="Low importance fact for user 1",
        user_id="user_alpha",
        importance_score=0.10,
    ))
    m_u2 = memory_manager.create_memory(MemoryCreate(
        content="Low importance fact for user 2",
        user_id="user_beta",
        importance_score=0.10,
    ))

    # Prune only user_alpha
    res = memory_manager.prune_memories(PruneRequest(
        user_id="user_alpha",
        importance_threshold=0.20,
    ))

    assert m_u1.id in res.pruned_memory_ids
    assert m_u2.id not in res.pruned_memory_ids

    # user_alpha memory is forgotten, user_beta memory remains active
    assert memory_manager.get_memory(m_u1.id).status == MemoryStatus.FORGOTTEN
    assert memory_manager.get_memory(m_u2.id).status == MemoryStatus.ACTIVE


def test_prune_api_endpoint(test_client: TestClient):
    """Test POST /api/v1/memories/prune endpoint via FastAPI TestClient."""
    # Create a test memory
    create_res = test_client.post("/api/v1/memories", json={
        "content": "Temporary endpoint test memory.",
        "user_id": "api_test_user",
        "importance_score": 0.10,
    })
    assert create_res.status_code == 201

    # Call /prune endpoint
    prune_res = test_client.post("/api/v1/memories/prune", json={
        "user_id": "api_test_user",
        "importance_threshold": 0.20,
        "hard_delete": False,
    })
    assert prune_res.status_code == 200
    data = prune_res.json()
    assert "total_evaluated" in data
    assert "pruned_count" in data
    assert "pruned_memory_ids" in data
    assert data["pruned_count"] >= 1
