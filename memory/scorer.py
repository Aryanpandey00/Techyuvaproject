"""
Memory Scorer: Implements the weighted re-ranking formula:
final_score = 0.50 * similarity + 0.20 * importance + 0.15 * recency + 0.15 * access_frequency
"""
from datetime import datetime, timezone
import logging
import math
from typing import Optional, Tuple
from core.schemas import Memory

logger = logging.getLogger(__name__)


class MemoryScorer:
    """
    Computes composite re-ranking score using configurable weights:
    - Semantic similarity
    - Memory intrinsic importance
    - Temporal recency decay
    - Access frequency score
    """

    def __init__(
        self,
        similarity_weight: float = 0.50,
        importance_weight: float = 0.20,
        recency_weight: float = 0.15,
        access_frequency_weight: float = 0.15,
        decay_rate_per_day: float = 0.05,
    ) -> None:
        self.similarity_weight = similarity_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.access_frequency_weight = access_frequency_weight
        self.decay_rate = decay_rate_per_day

    def calculate_recency_score(self, memory: Memory, now: Optional[datetime] = None) -> float:
        """
        Calculate temporal recency score [0.0 - 1.0] using exponential decay.
        Formula: exp(-decay_rate * delta_days)
        """
        current = now or datetime.now(timezone.utc)
        target_time = memory.updated_at or memory.created_at
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        delta_days = max(0.0, (current - target_time).total_seconds() / 86400.0)
        recency = math.exp(-self.decay_rate * delta_days)
        return max(0.0, min(1.0, recency))

    def calculate_decayed_importance(
        self,
        memory: Memory,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Calculate time-decayed importance score for memory retention / pruning.
        Formula: importance_score * exp(-decay_rate * delta_days)
        """
        recency = self.calculate_recency_score(memory, now=now)
        decayed = float(memory.importance_score) * recency
        return max(0.0, min(1.0, round(decayed, 4)))

    def calculate_access_frequency_score(self, memory: Memory) -> float:
        """
        Calculate access frequency score [0.0 - 1.0].
        Uses smooth diminishing returns scaling: tanh(0.2 * access_count)
        """
        count = max(0, memory.access_count)
        if count == 0:
            return 0.0
        # tanh(0.2 * 1) ~= 0.197, tanh(0.2 * 5) ~= 0.76, tanh(0.2 * 10) ~= 0.96
        score = math.tanh(0.2 * count)
        return max(0.0, min(1.0, score))

    def calculate_final_score(
        self,
        similarity: float,
        memory: Memory,
        now: Optional[datetime] = None,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute weighted final re-ranked score.
        Returns: (final_score, similarity, importance, recency, access_frequency)
        """
        sim_val = max(0.0, min(1.0, float(similarity)))
        imp_val = max(0.0, min(1.0, float(memory.importance_score)))
        rec_val = self.calculate_recency_score(memory, now)
        acc_val = self.calculate_access_frequency_score(memory)

        final = (
            self.similarity_weight * sim_val
            + self.importance_weight * imp_val
            + self.recency_weight * rec_val
            + self.access_frequency_weight * acc_val
        )

        final_score = max(0.0, min(1.0, final))
        return (
            round(final_score, 4),
            round(sim_val, 4),
            round(imp_val, 4),
            round(rec_val, 4),
            round(acc_val, 4),
        )
