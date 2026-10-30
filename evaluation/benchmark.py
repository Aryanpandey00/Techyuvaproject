"""
Evaluation Benchmark Runner: evaluates context size reduction, retrieval latency, and memory fidelity.
"""
import time
from typing import Any, Dict, List
from core.schemas import CompressionRequest, MemoryCreate, MemoryRetrieveRequest, MemoryType
from evaluation.metrics import (
    calculate_compression_ratio,
    calculate_redundancy_index,
    calculate_token_savings,
)
from memory.manager import MemoryManager


class MemoryBenchmarkRunner:
    """
    Runs automated comparative experiments comparing uncompressed raw memories
    versus compressed semantic memory representation.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.manager = memory_manager

    def load_synthetic_benchmark_dataset(self) -> List[MemoryCreate]:
        """Generate a realistic set of redundant and diverse conversation memories."""
        return [
            # Cluster 1: Python programming preferences
            MemoryCreate(content="User prefers Python for backend development and ML.", memory_type=MemoryType.PREFERENCE, importance_score=0.9),
            MemoryCreate(content="User likes using FastAPI for creating REST APIs in Python.", memory_type=MemoryType.PREFERENCE, importance_score=0.85),
            MemoryCreate(content="User works heavily with PyTorch and Sentence Transformers in Python.", memory_type=MemoryType.FACT, importance_score=0.88),
            
            # Cluster 2: Geographic & Personal
            MemoryCreate(content="User lives in Bangalore, India.", memory_type=MemoryType.FACT, importance_score=0.95),
            MemoryCreate(content="User's current time zone is IST (+05:30) in Bangalore.", memory_type=MemoryType.FACT, importance_score=0.70),
            MemoryCreate(content="User works remotely from Bangalore as an AI engineer.", memory_type=MemoryType.FACT, importance_score=0.90),

            # Standalone items
            MemoryCreate(content="User is building a final year project on AI Memory Compression.", memory_type=MemoryType.FACT, importance_score=0.98),
            MemoryCreate(content="User prefers dark mode UI themes with modern gradients.", memory_type=MemoryType.PREFERENCE, importance_score=0.65),
        ]

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Execute end-to-end benchmark:
        1. Seed database with synthetic memory statements
        2. Compute baseline metrics (redundancy, token usage)
        3. Execute semantic compression
        4. Measure compression ratio, token savings, latency
        5. Return comprehensive diagnostic summary
        """
        # 1. Seed dataset
        items = self.load_synthetic_benchmark_dataset()
        created_memories = []
        for item in items:
            created = self.manager.create_memory(item)
            created_memories.append(created)

        original_texts = [m.content for m in created_memories]
        
        # Calculate baseline embeddings and redundancy
        orig_embeddings = self.manager.encoder.encode_batch(original_texts)
        pre_redundancy = calculate_redundancy_index(orig_embeddings)

        # 2. Benchmark Retrieval Latency (Pre-compression)
        t0 = time.perf_counter()
        pre_retrieve = self.manager.retrieve_context(
            MemoryRetrieveRequest(query="What programming language and frameworks do I use?", top_k=5)
        )
        pre_retrieve_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Run Compression
        t_comp_0 = time.perf_counter()
        compression_result = self.manager.compress_memories(
            CompressionRequest(similarity_threshold=0.65, min_cluster_size=2)
        )
        compression_duration_ms = round((time.perf_counter() - t_comp_0) * 1000, 2)

        # 4. Benchmark Retrieval Latency (Post-compression)
        t1 = time.perf_counter()
        post_retrieve = self.manager.retrieve_context(
            MemoryRetrieveRequest(query="What programming language and frameworks do I use?", top_k=5)
        )
        post_retrieve_latency_ms = round((time.perf_counter() - t1) * 1000, 2)

        # Gather active memories after compression
        post_active = self.manager.list_memories(limit=100)
        post_texts = [m.content for m in post_active]
        
        # 5. Compute Post-compression Metrics
        comp_ratio = calculate_compression_ratio(original_texts, post_texts)
        token_stats = calculate_token_savings(original_texts, post_texts)

        return {
            "benchmark_status": "SUCCESS",
            "initial_memory_count": len(created_memories),
            "clusters_compressed": compression_result.clusters_found,
            "memories_merged": compression_result.memories_compressed_count,
            "new_summaries_generated": compression_result.new_summaries_created,
            "final_active_memory_count": len(post_texts),
            "compression_ratio": comp_ratio,
            "token_stats": token_stats,
            "redundancy_index_before": pre_redundancy,
            "compression_time_ms": compression_duration_ms,
            "retrieval_latency_ms_before": pre_retrieve_latency_ms,
            "retrieval_latency_ms_after": post_retrieve_latency_ms,
        }

    def run_retrieval_benchmark(self, user_id: str = "eval_bench_user") -> Dict[str, Any]:
        """
        Run retrieval benchmark evaluating Precision@K, Recall@K, success rate, and distractor filtering.
        Uses canonical Part C benchmark dataset:
        1. User preference: "My favorite programming language is Python."
        2. Project context: "I am building an AI memory system."
        3. Temporary info: "I had tea this morning."
        """
        from evaluation.metrics import (
            calculate_irrelevant_filter_rate,
            calculate_retrieval_precision_at_k,
            calculate_retrieval_recall_at_k,
            calculate_retrieval_success_rate,
        )

        # 1. Seed canonical memories
        m_pref = self.manager.create_memory(MemoryCreate(
            content="My favorite programming language is Python.",
            user_id=user_id,
            memory_type=MemoryType.PREFERENCE,
            importance_score=0.90,
        ))
        m_proj = self.manager.create_memory(MemoryCreate(
            content="I am building an AI memory system.",
            user_id=user_id,
            memory_type=MemoryType.PROJECT,
            importance_score=0.95,
        ))
        m_temp = self.manager.create_memory(MemoryCreate(
            content="I had tea this morning.",
            user_id=user_id,
            memory_type=MemoryType.TEMPORARY,
            importance_score=0.15,
        ))

        test_cases = [
            {
                "query": "What programming language do I like?",
                "expected_ids": [m_pref.id],
                "irrelevant_ids": [m_temp.id],
            },
            {
                "query": "What project am I building?",
                "expected_ids": [m_proj.id],
                "irrelevant_ids": [m_temp.id],
            },
            {
                "query": "What is my favorite language?",
                "expected_ids": [m_pref.id],
                "irrelevant_ids": [m_temp.id],
            },
        ]

        query_hits: List[bool] = []
        precisions: List[float] = []
        recalls: List[float] = []
        filter_rates: List[float] = []
        case_diagnostics = []

        for case in test_cases:
            resp = self.manager.retrieve_context(MemoryRetrieveRequest(
                query=case["query"],
                user_id=user_id,
                top_k=2,
            ))
            retrieved_ids = [r.memory.id for r in resp.retrieved_memories]

            p_k = calculate_retrieval_precision_at_k(retrieved_ids, case["expected_ids"], k=2)
            r_k = calculate_retrieval_recall_at_k(retrieved_ids, case["expected_ids"], k=2)
            f_rate = calculate_irrelevant_filter_rate(retrieved_ids, case["irrelevant_ids"])
            has_hit = any(eid in retrieved_ids for eid in case["expected_ids"])

            query_hits.append(has_hit)
            precisions.append(p_k)
            recalls.append(r_k)
            filter_rates.append(f_rate)

            case_diagnostics.append({
                "query": case["query"],
                "retrieved_count": len(retrieved_ids),
                "precision_at_k": p_k,
                "recall_at_k": r_k,
                "irrelevant_filter_rate": f_rate,
                "retrieved_contents": [r.memory.content for r in resp.retrieved_memories],
            })

        mean_precision = round(sum(precisions) / float(len(precisions)), 4) if precisions else 0.0
        mean_recall = round(sum(recalls) / float(len(recalls)), 4) if recalls else 0.0
        mean_filter_rate = round(sum(filter_rates) / float(len(filter_rates)), 4) if filter_rates else 1.0
        overall_success_rate = calculate_retrieval_success_rate(query_hits)

        return {
            "retrieval_benchmark_status": "SUCCESS",
            "total_queries_evaluated": len(test_cases),
            "retrieval_success_rate": overall_success_rate,
            "mean_precision_at_k": mean_precision,
            "mean_recall_at_k": mean_recall,
            "mean_irrelevant_filter_rate": mean_filter_rate,
            "query_diagnostics": case_diagnostics,
            "seeded_memory_ids": {
                "preference": m_pref.id,
                "project": m_proj.id,
                "temporary": m_temp.id,
            },
        }

    def run_retention_benchmark(self, user_id: str = "retention_bench_user") -> Dict[str, Any]:
        """
        Evaluate memory retention and forgetting:
        - Pinned & high-importance memories are protected.
        - Decayed low-importance memories are forgotten.
        """
        from datetime import datetime, timezone, timedelta
        from core.schemas import PruneRequest
        from evaluation.metrics import calculate_retention_accuracy

        now = datetime.now(timezone.utc)
        eighty_days_ago = now - timedelta(days=80)

        # 1. Seed memories
        m_pinned = self.manager.create_memory(MemoryCreate(
            content="Critical user rule: never share passwords.",
            user_id=user_id,
            importance_score=0.20,
            metadata={"is_pinned": True},
        ))
        m_high = self.manager.create_memory(MemoryCreate(
            content="User has severe allergy to peanuts.",
            user_id=user_id,
            importance_score=0.95,
        ))
        m_decayed = self.manager.create_memory(MemoryCreate(
            content="Temporary meeting room code is 5849.",
            user_id=user_id,
            importance_score=0.15,
        ))

        # Artificially age the decayed memory
        with self.manager.sqlite_store._get_connection() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
                (eighty_days_ago.isoformat(), eighty_days_ago.isoformat(), eighty_days_ago.isoformat(), m_decayed.id)
            )
            conn.commit()

        expected_retained = [m_pinned.id, m_high.id]
        expected_forgotten = [m_decayed.id]

        # 2. Run pruning pass
        prune_result = self.manager.prune_memories(PruneRequest(
            user_id=user_id,
            importance_threshold=0.20,
            hard_delete=False,
        ))

        active_memories = self.manager.list_memories(user_id=user_id)
        active_ids = [m.id for m in active_memories]

        retention_metrics = calculate_retention_accuracy(
            retained_actual_ids=active_ids,
            expected_retained_ids=expected_retained,
            forgotten_actual_ids=prune_result.pruned_memory_ids,
            expected_forgotten_ids=expected_forgotten,
        )

        return {
            "retention_benchmark_status": "SUCCESS",
            "total_evaluated": prune_result.total_evaluated,
            "pruned_count": prune_result.pruned_count,
            "retention_metrics": retention_metrics,
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Execute unified benchmark testing compression, retrieval relevance/filtering, and retention.
        """
        comp_report = self.run_full_benchmark()
        retrieval_report = self.run_retrieval_benchmark()
        retention_report = self.run_retention_benchmark()

        return {
            "overall_status": "SUCCESS",
            "compression_benchmark": comp_report,
            "retrieval_benchmark": retrieval_report,
            "retention_benchmark": retention_report,
        }
