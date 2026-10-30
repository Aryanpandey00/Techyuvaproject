"""
Tests for memory creation, updating, deletion, and extraction.
"""
from datetime import datetime, timezone, timedelta
from core.schemas import Memory, MemoryCreate, MemoryStatus, MemoryType, MemoryUpdate
from memory.manager import MemoryManager
from memory.scorer import MemoryScorer


def test_create_and_get_memory(memory_manager: MemoryManager):
    """Test creating a memory item and fetching it by ID."""
    req = MemoryCreate(
        content="I prefer dark roasted coffee with oat milk.",
        user_id="user_1",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.85,
    )
    created = memory_manager.create_memory(req)
    assert created.id is not None
    assert created.content == req.content
    assert created.status == MemoryStatus.ACTIVE
    assert created.importance_score == 0.85

    fetched = memory_manager.get_memory(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.content == req.content


def test_update_memory(memory_manager: MemoryManager):
    """Test updating memory content and lifecycle status."""
    created = memory_manager.create_memory(MemoryCreate(
        content="Initial fact content",
        user_id="user_1",
        memory_type=MemoryType.FACT,
        importance_score=0.5
    ))

    update_payload = MemoryUpdate(
        content="Updated fact content",
        importance_score=0.95,
        status=MemoryStatus.ARCHIVED
    )
    updated = memory_manager.update_memory(created.id, update_payload)
    assert updated.content == "Updated fact content"
    assert updated.importance_score == 0.95
    assert updated.status == MemoryStatus.ARCHIVED

    refetched = memory_manager.get_memory(created.id)
    assert refetched.content == "Updated fact content"
    assert refetched.status == MemoryStatus.ARCHIVED


def test_delete_memory(memory_manager: MemoryManager):
    """Test deleting memory permanently."""
    created = memory_manager.create_memory(MemoryCreate(content="Temporary memory to delete"))
    assert memory_manager.get_memory(created.id) is not None

    deleted = memory_manager.delete_memory(created.id)
    assert deleted is True
    assert memory_manager.get_memory(created.id) is None


def test_memory_scorer_decay():
    """Test temporal decay calculation in MemoryScorer."""
    scorer = MemoryScorer(decay_rate_per_day=0.05)
    now = datetime.now(timezone.utc)
    ten_days_ago = now - timedelta(days=10)

    mem_new = MemoryCreate(content="New fact", created_at=now).model_dump()
    mem_new_obj = Memory(**mem_new)

    mem_old = MemoryCreate(content="Old fact").model_dump()
    mem_old["updated_at"] = ten_days_ago
    mem_old["created_at"] = ten_days_ago
    mem_old_obj = Memory(**mem_old)

    recency_new = scorer.calculate_recency_score(mem_new_obj, now)
    recency_old = scorer.calculate_recency_score(mem_old_obj, now)

    assert recency_new == 1.0
    assert recency_old < 1.0
    assert recency_old > 0.0


def test_memory_extraction(memory_manager: MemoryManager):
    """Test automatic extraction from user statement."""
    extracted = memory_manager.extractor.extract_from_turn(
        user_message="My name is Alex and I live in Seattle."
    )
    assert len(extracted) >= 1
    assert any("Alex" in m.content or "Seattle" in m.content for m in extracted)
