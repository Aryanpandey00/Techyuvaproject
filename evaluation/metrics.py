"""
Quantitative evaluation metrics for memory compression, token savings, redundancy, and retrieval.
"""
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_compression_ratio(
    original_texts: List[str],
    compressed_texts: List[str],
) -> float:
    """
    Calculate text character compression ratio:
    ratio = len(compressed_chars) / len(original_chars)
    Values < 1.0 indicate successful size reduction.
    """
    orig_len = sum(len(t) for t in original_texts)
    comp_len = sum(len(t) for t in compressed_texts)
    if orig_len == 0:
        return 1.0
    return round(comp_len / orig_len, 4)


def calculate_token_savings(
    original_texts: List[str],
    compressed_texts: List[str],
) -> Dict[str, Any]:
    """
    Estimate token count difference (approximation: ~4 chars per token) and percent savings.
    """
    orig_chars = sum(len(t) for t in original_texts)
    comp_chars = sum(len(t) for t in compressed_texts)

    orig_tokens = max(1, orig_chars // 4)
    comp_tokens = max(0, comp_chars // 4)
    saved_tokens = max(0, orig_tokens - comp_tokens)
    pct_saved = round((saved_tokens / orig_tokens) * 100.0, 2)

    return {
        "original_tokens_est": orig_tokens,
        "compressed_tokens_est": comp_tokens,
        "tokens_saved": saved_tokens,
        "percentage_saved": pct_saved,
    }


def calculate_redundancy_index(embeddings: List[List[float]]) -> float:
    """
    Calculate average pairwise cosine similarity of memory embeddings.
    A higher score indicates greater semantic overlap/redundancy.
    """
    if len(embeddings) < 2:
        return 0.0
    arr = np.array(embeddings)
    sim_mat = cosine_similarity(arr)
    # Take upper triangle excluding diagonal
    n = len(embeddings)
    triu_indices = np.triu_indices(n, k=1)
    return round(float(np.mean(sim_mat[triu_indices])), 4)


def calculate_retrieval_precision_at_k(
    retrieved_ids: List[str],
    ground_truth_relevant_ids: List[str],
    k: int = 5,
) -> float:
    """Precision@K = (# relevant items in top K) / K"""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(ground_truth_relevant_ids)
    hits = sum(1 for item in top_k if item in relevant_set)
    return round(hits / float(k), 4)


def calculate_retrieval_recall_at_k(
    retrieved_ids: List[str],
    ground_truth_relevant_ids: List[str],
    k: int = 5,
) -> float:
    """Recall@K = (# relevant items in top K) / (total relevant items)"""
    if not ground_truth_relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    relevant_set = set(ground_truth_relevant_ids)
    hits = sum(1 for item in top_k if item in relevant_set)
    return round(hits / float(len(relevant_set)), 4)


def calculate_retrieval_success_rate(query_hits: List[bool]) -> float:
    """
    Calculate overall retrieval success rate across a batch of test queries:
    success_rate = (number of queries with >= 1 relevant hit) / (total queries)
    """
    if not query_hits:
        return 0.0
    successful = sum(1 for hit in query_hits if hit)
    return round(successful / float(len(query_hits)), 4)


def calculate_irrelevant_filter_rate(
    retrieved_ids: List[str],
    irrelevant_ids: List[str],
) -> float:
    """
    Calculate the rate at which irrelevant distractors are successfully filtered out:
    filter_rate = (# irrelevant items NOT in retrieved) / (total irrelevant items)
    """
    if not irrelevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    unretrieved_distractors = sum(1 for item in irrelevant_ids if item not in retrieved_set)
    return round(unretrieved_distractors / float(len(irrelevant_ids)), 4)


def calculate_retention_accuracy(
    retained_actual_ids: List[str],
    expected_retained_ids: List[str],
    forgotten_actual_ids: List[str],
    expected_forgotten_ids: List[str],
) -> Dict[str, float]:
    """
    Calculate retention rate of high-importance/pinned memories
    and forgetting rate of low-value obsolete memories.
    """
    retained_set = set(retained_actual_ids)
    forgotten_set = set(forgotten_actual_ids)

    retention_hits = sum(1 for mid in expected_retained_ids if mid in retained_set)
    retention_rate = round(retention_hits / float(len(expected_retained_ids)), 4) if expected_retained_ids else 1.0

    forgetting_hits = sum(1 for mid in expected_forgotten_ids if mid in forgotten_set)
    forgetting_rate = round(forgetting_hits / float(len(expected_forgotten_ids)), 4) if expected_forgotten_ids else 1.0

    overall_accuracy = round((retention_rate + forgetting_rate) / 2.0, 4)

    return {
        "retention_rate": retention_rate,
        "forgetting_rate": forgetting_rate,
        "overall_accuracy": overall_accuracy,
    }


def calculate_context_efficiency(
    original_tokens: int,
    injected_tokens: int,
) -> Dict[str, Any]:
    """
    Calculate context reduction and token savings between full uncompressed history
    and the resource-efficient injected context.
    """
    orig = max(1, original_tokens)
    injected = max(0, injected_tokens)
    saved = max(0, orig - injected)
    pct_reduction = round((saved / float(orig)) * 100.0, 2)

    return {
        "original_tokens_est": orig,
        "injected_tokens_est": injected,
        "tokens_saved_est": saved,
        "percentage_reduction": pct_reduction,
    }


def classify_candidate_relevance(score: float, threshold: float = 0.65) -> str:
    """
    Classify memory candidate into RELEVANT, WEAKLY_RELATED, or IRRELEVANT.
    """
    if score >= threshold:
        return "relevant"
    elif score >= max(0.30, threshold - 0.25):
        return "weakly_related"
    else:
        return "irrelevant"
