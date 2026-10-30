"""
Evaluation package containing quantitative metrics and benchmark runner for memory compression.
"""
from evaluation.metrics import (
    calculate_compression_ratio,
    calculate_token_savings,
    calculate_redundancy_index,
    calculate_retrieval_precision_at_k,
    calculate_retrieval_recall_at_k,
    calculate_retrieval_success_rate,
    calculate_irrelevant_filter_rate,
    calculate_retention_accuracy,
    calculate_context_efficiency,
    classify_candidate_relevance,
)
from evaluation.benchmark import MemoryBenchmarkRunner

__all__ = [
    "calculate_compression_ratio",
    "calculate_token_savings",
    "calculate_redundancy_index",
    "calculate_retrieval_precision_at_k",
    "calculate_retrieval_recall_at_k",
    "calculate_retrieval_success_rate",
    "calculate_irrelevant_filter_rate",
    "calculate_retention_accuracy",
    "calculate_context_efficiency",
    "classify_candidate_relevance",
    "MemoryBenchmarkRunner",
]
