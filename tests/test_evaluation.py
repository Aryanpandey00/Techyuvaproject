"""
Unit and integration tests for Evaluation Metrics and Memory Benchmark Suite:
- Quantitative metrics: Precision@K, Recall@K, Success Rate, Distractor Filtering, Compression Ratio, Token Savings, Redundancy
- Retrieval relevance and Top-K ranking
- Irrelevant-memory filtering
- Part C Canonical Benchmark Scenario
- Compression effectiveness
- Memory retention and forgetting evaluation
- MemoryBenchmarkRunner execution
"""
import pytest

from core.schemas import MemoryCreate, MemoryRetrieveRequest, MemoryType
from evaluation.benchmark import MemoryBenchmarkRunner
from evaluation.metrics import (
    calculate_compression_ratio,
    calculate_irrelevant_filter_rate,
    calculate_redundancy_index,
    calculate_retention_accuracy,
    calculate_retrieval_precision_at_k,
    calculate_retrieval_recall_at_k,
    calculate_retrieval_success_rate,
    calculate_token_savings,
)
from memory.manager import MemoryManager


def test_evaluation_metric_functions():
    """Verify standalone metric calculation functions with controlled inputs."""
    # 1. Precision@K and Recall@K
    retrieved = ["id_1", "id_2", "id_3", "id_4", "id_5"]
    ground_truth = ["id_2", "id_5", "id_9"]

    # In top 3: ["id_1", "id_2", "id_3"] -> 1 hit ("id_2")
    p_3 = calculate_retrieval_precision_at_k(retrieved, ground_truth, k=3)
    assert p_3 == round(1 / 3.0, 4)

    # In top 5: ["id_1", "id_2", "id_3", "id_4", "id_5"] -> 2 hits ("id_2", "id_5")
    p_5 = calculate_retrieval_precision_at_k(retrieved, ground_truth, k=5)
    assert p_5 == 0.4

    r_5 = calculate_retrieval_recall_at_k(retrieved, ground_truth, k=5)
    assert r_5 == round(2 / 3.0, 4)

    # 2. Retrieval success rate
    hits = [True, False, True, True]
    success_rate = calculate_retrieval_success_rate(hits)
    assert success_rate == 0.75

    # 3. Irrelevant filter rate
    irrelevant = ["noise_1", "noise_2", "noise_3"]
    # Retrieved has "id_1", "noise_1" -> 1 noise leaked, 2 filtered
    filter_rate = calculate_irrelevant_filter_rate(["id_1", "noise_1"], irrelevant)
    assert filter_rate == round(2 / 3.0, 4)

    # 4. Compression ratio and token savings
    orig = ["Long text memory detailing user daily workflow.", "Second verbose memory about project setup."]
    comp = ["Merged compact memory."]
    ratio = calculate_compression_ratio(orig, comp)
    assert 0.0 < ratio < 1.0

    tok = calculate_token_savings(orig, comp)
    assert tok["tokens_saved"] > 0
    assert tok["percentage_saved"] > 0.0

    # 5. Redundancy index
    # Identity embeddings (100% duplicate) -> redundancy = 1.0
    identical_embs = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    red = calculate_redundancy_index(identical_embs)
    assert red == 1.0


def test_part_c_canonical_benchmark(memory_manager: MemoryManager):
    """
    Execute the Part C canonical benchmark:
    User preference: "My favorite programming language is Python."
    Project context: "I am building an AI memory system."
    Temporary information: "I had tea this morning."
    
    Test queries:
    - "What programming language do I like?"
    - "What project am I building?"
    - "What is my favorite language?"
    
    Verify that relevant memories are retrieved while irrelevant memories are filtered.
    """
    user_id = "part_c_user"

    # Seed the exact conversations specified in requirements
    m_pref = memory_manager.create_memory(MemoryCreate(
        content="My favorite programming language is Python.",
        user_id=user_id,
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.90,
    ))
    m_proj = memory_manager.create_memory(MemoryCreate(
        content="I am building an AI memory system.",
        user_id=user_id,
        memory_type=MemoryType.PROJECT,
        importance_score=0.95,
    ))
    m_temp = memory_manager.create_memory(MemoryCreate(
        content="I had tea this morning.",
        user_id=user_id,
        memory_type=MemoryType.TEMPORARY,
        importance_score=0.15,
    ))

    # Query 1: Programming language preference
    resp1 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="What programming language do I like?",
        user_id=user_id,
        top_k=2,
    ))
    retrieved_ids1 = [r.memory.id for r in resp1.retrieved_memories]
    assert m_pref.id in retrieved_ids1, "Expected preference memory to be retrieved"
    assert m_temp.id not in retrieved_ids1, "Expected temporary tea memory to be filtered out"

    # Query 2: Current project
    resp2 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="What project am I building?",
        user_id=user_id,
        top_k=2,
    ))
    retrieved_ids2 = [r.memory.id for r in resp2.retrieved_memories]
    assert m_proj.id in retrieved_ids2, "Expected project memory to be retrieved"
    assert m_temp.id not in retrieved_ids2, "Expected temporary tea memory to be filtered out"

    # Query 3: Favorite language
    resp3 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="What is my favorite language?",
        user_id=user_id,
        top_k=2,
    ))
    retrieved_ids3 = [r.memory.id for r in resp3.retrieved_memories]
    assert m_pref.id in retrieved_ids3, "Expected preference memory to be retrieved"
    assert m_temp.id not in retrieved_ids3, "Expected temporary tea memory to be filtered out"


def test_top_k_retrieval_behavior(memory_manager: MemoryManager):
    """Verify Top-K truncation and composite ranking behavior during evaluation."""
    user_id = "top_k_user"

    # Seed 5 memories on related topic
    for i in range(5):
        memory_manager.create_memory(MemoryCreate(
            content=f"Database indexing strategy variation {i}: B-tree and hash indexes.",
            user_id=user_id,
            importance_score=0.5 + 0.1 * i,
        ))

    # Top-K = 2
    resp_k2 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="database indexing",
        user_id=user_id,
        top_k=2,
    ))
    assert len(resp_k2.retrieved_memories) == 2
    assert resp_k2.retrieved_memories[0].final_score >= resp_k2.retrieved_memories[1].final_score

    # Top-K = 4
    resp_k4 = memory_manager.retrieve_context(MemoryRetrieveRequest(
        query="database indexing",
        user_id=user_id,
        top_k=4,
    ))
    assert len(resp_k4.retrieved_memories) == 4


def test_runner_retrieval_benchmark(memory_manager: MemoryManager):
    """Verify MemoryBenchmarkRunner.run_retrieval_benchmark returns complete metrics."""
    runner = MemoryBenchmarkRunner(memory_manager)
    report = runner.run_retrieval_benchmark(user_id="runner_bench_user")

    assert report["retrieval_benchmark_status"] == "SUCCESS"
    assert report["total_queries_evaluated"] == 3
    assert report["retrieval_success_rate"] == 1.0
    assert report["mean_precision_at_k"] > 0.0
    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_irrelevant_filter_rate"] == 1.0
    assert len(report["query_diagnostics"]) == 3


def test_runner_retention_benchmark(memory_manager: MemoryManager):
    """Verify MemoryBenchmarkRunner.run_retention_benchmark evaluates retention accuracy."""
    runner = MemoryBenchmarkRunner(memory_manager)
    report = runner.run_retention_benchmark(user_id="retention_test_user")

    assert report["retention_benchmark_status"] == "SUCCESS"
    assert report["total_evaluated"] >= 3
    assert report["pruned_count"] >= 1
    ret_metrics = report["retention_metrics"]
    assert ret_metrics["retention_rate"] == 1.0
    assert ret_metrics["forgetting_rate"] == 1.0
    assert ret_metrics["overall_accuracy"] == 1.0


def test_runner_full_and_comprehensive_benchmark(memory_manager: MemoryManager):
    """Verify end-to-end benchmark execution including compression and diagnostics."""
    runner = MemoryBenchmarkRunner(memory_manager)
    comp_report = runner.run_comprehensive_benchmark()

    assert comp_report["overall_status"] == "SUCCESS"
    assert "compression_benchmark" in comp_report
    assert "retrieval_benchmark" in comp_report
    assert "retention_benchmark" in comp_report
    assert comp_report["compression_benchmark"]["benchmark_status"] == "SUCCESS"
    assert comp_report["compression_benchmark"]["initial_memory_count"] > 0
