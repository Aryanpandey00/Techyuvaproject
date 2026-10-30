"""
Unit tests for AI Memory Compression:
- Clustering semantically similar memories
- Generating concise compressed memories via LLM
- Preserving important facts and maintaining parent ID references
- Updating original memory status to 'compressed'
- Calculating new importance score
- Quantitative compression statistics (ratio, size, retention, token savings)
- Configurable maximum memory length
- Mocked LLM response verification
- compress_memories(memory_ids) function interface
"""
from unittest.mock import MagicMock
import pytest
from core.schemas import CompressionRequest, MemoryCreate, MemoryStatus, MemoryType
from database.memory_storage import MemoryStorageService
from embeddings.encoder import MockEmbeddingEncoder
from llm.base import BaseLLMProvider
from memory.compressor import MemoryCompressor, compress_memories
from memory.manager import MemoryManager


def test_compress_memories_python_example(memory_manager: MemoryManager):
    """
    Test the exact prompt example:
    Memory 1: 'I am learning Python.'
    Memory 2: 'I use Python for college projects.'
    Memory 3: 'Python is my preferred programming language.'
    Memory 4: 'I am currently building projects using Python.'
    Expected: 'User is learning and prefers Python and frequently uses it for college projects.'
    """
    m1 = memory_manager.create_memory(MemoryCreate(
        content="I am learning Python.",
        user_id="rahul",
        memory_type=MemoryType.FACT,
        importance_score=0.80,
    ))
    m2 = memory_manager.create_memory(MemoryCreate(
        content="I use Python for college projects.",
        user_id="rahul",
        memory_type=MemoryType.PROJECT,
        importance_score=0.85,
    ))
    m3 = memory_manager.create_memory(MemoryCreate(
        content="Python is my preferred programming language.",
        user_id="rahul",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.90,
    ))
    m4 = memory_manager.create_memory(MemoryCreate(
        content="I am currently building projects using Python.",
        user_id="rahul",
        memory_type=MemoryType.PROJECT,
        importance_score=0.85,
    ))

    # Compress by explicit IDs
    result = compress_memories(
        memory_ids=[m1.id, m2.id, m3.id, m4.id],
        user_id="rahul",
        storage=memory_manager.storage,
        llm=memory_manager.llm,
        encoder=memory_manager.encoder,
    )

    assert result.compressed_memory is not None
    comp = result.compressed_memory

    # 1. Content check
    assert "Python" in comp.content
    assert "User" in comp.content or "learning" in comp.content or "projects" in comp.content

    # 2. Lineage: parent_ids check
    assert len(comp.parent_ids) == 4
    for orig_id in [m1.id, m2.id, m3.id, m4.id]:
        assert orig_id in comp.parent_ids

    # 3. New importance score calculated
    assert comp.importance_score >= 0.90

    # 4. Original memories marked as compressed in SQLite
    for orig_id in [m1.id, m2.id, m3.id, m4.id]:
        refetched = memory_manager.get_memory(orig_id)
        assert refetched is not None
        assert refetched.status == MemoryStatus.COMPRESSED

    # 5. Compression statistics
    assert result.stats is not None
    stats = result.stats
    assert stats.original_memory_count == 4
    assert stats.compressed_memory_size == len(comp.content)
    assert 0.0 < stats.compression_ratio < 1.0
    assert stats.information_retained >= 0.90
    assert stats.tokens_saved_est >= 0


def test_compress_memories_function_interface(memory_manager: MemoryManager):
    """Test compress_memories(memory_ids) top-level convenience function."""
    m1 = memory_manager.create_memory(MemoryCreate(
        content="User lives in Bangalore.",
        user_id="dev_user",
        importance_score=0.9,
    ))
    m2 = memory_manager.create_memory(MemoryCreate(
        content="User works in Bangalore as an AI engineer.",
        user_id="dev_user",
        importance_score=0.95,
    ))

    result = compress_memories(
        memory_ids=[m1.id, m2.id],
        user_id="dev_user",
        storage=memory_manager.storage,
    )

    assert result.clusters_found == 1
    assert result.memories_compressed_count == 2
    assert result.compressed_memory is not None
    assert "Bangalore" in result.compressed_memory.content


def test_auto_clustering_compression(memory_manager: MemoryManager):
    """Test automatic semantic clustering and batch compression."""
    # Seed 3 related items
    m1 = memory_manager.create_memory(MemoryCreate(content="I build neural networks in PyTorch.", user_id="user_cluster"))
    m2 = memory_manager.create_memory(MemoryCreate(content="I train deep learning models using PyTorch.", user_id="user_cluster"))
    m3 = memory_manager.create_memory(MemoryCreate(content="PyTorch is my favorite deep learning framework.", user_id="user_cluster"))

    # Seed 1 unrelated item
    m_unrelated = memory_manager.create_memory(MemoryCreate(content="My favorite pizza topping is mushrooms.", user_id="user_cluster"))

    # Execute auto-clustering compression
    req = CompressionRequest(
        user_id="user_cluster",
        similarity_threshold=0.0,  # mock encoder clustering threshold
        min_cluster_size=2,
    )
    result = memory_manager.compress_memories(req)

    assert result.new_summaries_created >= 1
    assert len(result.compressed_items) >= 1


def test_max_memory_length_constraint(mock_encoder: MockEmbeddingEncoder):
    """Test that generated compressed memory obeys max_memory_length."""
    storage = MemoryStorageService(
        sqlite_db_path=":memory:",
        chroma_persist_dir=None,
        encoder=mock_encoder,
        in_memory=True,
    )

    m1 = storage.create_memory(MemoryCreate(content="First very long sentence with many details."))
    m2 = storage.create_memory(MemoryCreate(content="Second very long sentence with many details."))

    compressor = MemoryCompressor(
        storage_service=storage,
        encoder=mock_encoder,
        max_memory_length=50,
    )

    comp_mem, _, _ = compressor.compress_memory_cluster([m1, m2], max_memory_length=50)
    assert len(comp_mem.content) <= 50


def test_compression_with_mocked_llm(mock_encoder: MockEmbeddingEncoder):
    """Test MemoryCompressor with custom Mocked LLM provider."""
    storage = MemoryStorageService(
        sqlite_db_path=":memory:",
        chroma_persist_dir=None,
        encoder=mock_encoder,
        in_memory=True,
    )

    m1 = storage.create_memory(MemoryCreate(content="User likes Italian food.", importance_score=0.7))
    m2 = storage.create_memory(MemoryCreate(content="User favorite dish is pasta carbonara.", importance_score=0.85))

    mock_llm = MagicMock(spec=BaseLLMProvider)
    mock_llm.generate_response.return_value = "User prefers Italian cuisine and particularly enjoys pasta carbonara."

    compressor = MemoryCompressor(
        storage_service=storage,
        llm_provider=mock_llm,
        encoder=mock_encoder,
    )

    comp_mem, orig_ids, stats = compressor.compress_memory_cluster([m1, m2])

    assert comp_mem.content == "User prefers Italian cuisine and particularly enjoys pasta carbonara."
    assert comp_mem.importance_score >= 0.85
    assert len(orig_ids) == 2
    assert stats.original_memory_count == 2
    assert stats.compression_ratio > 0.0


def test_dry_run_compression(memory_manager: MemoryManager):
    """Test that dry_run compression does not mutate database records."""
    m1 = memory_manager.create_memory(MemoryCreate(content="Fact Alpha for dry run."))
    m2 = memory_manager.create_memory(MemoryCreate(content="Fact Beta for dry run."))

    req = CompressionRequest(
        memory_ids=[m1.id, m2.id],
        dry_run=True,
    )
    result = memory_manager.compress_memories(req)

    assert result.compressed_memory is not None

    # Verify original memories are STILL active
    refetched_1 = memory_manager.get_memory(m1.id)
    refetched_2 = memory_manager.get_memory(m2.id)
    assert refetched_1.status == MemoryStatus.ACTIVE
    assert refetched_2.status == MemoryStatus.ACTIVE
