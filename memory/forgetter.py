"""
Memory Forgetter: executes decay-based pruning and archiving of obsolete or low-utility memories.
"""
from datetime import datetime, timezone
import logging
from typing import List, Optional
from core.schemas import MemoryItem, MemoryStatus, PruneRequest, PruneResult
from database.chroma_store import ChromaVectorStore
from database.sqlite_store import SQLiteMetadataStore
from memory.scorer import MemoryScorer

logger = logging.getLogger(__name__)


class MemoryForgetter:
    """
    Evaluates memory longevity and utility, pruning memories whose decayed importance
    drops below a defined threshold or whose inactivity exceeds limits.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        metadata_store: SQLiteMetadataStore,
        scorer: Optional[MemoryScorer] = None,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.scorer = scorer or MemoryScorer()

    def decide_lifecycle(
        self,
        memory: MemoryItem,
        threshold: float = 0.20,
        max_inactivity_days: Optional[int] = 30,
        now: Optional[datetime] = None,
    ) -> str:
        """
        Evaluate memory lifecycle stage and return one of:
        - 'KEEP': Active, high-importance, or pinned memory
        - 'ARCHIVE': Stale or low-frequency memory that should be archived
        - 'FORGET': Decayed or inactive low-value memory that should be forgotten/pruned
        """
        current_time = now or datetime.now(timezone.utc)

        # 1. Do not prune protected or pinned memories
        is_pinned = False
        if hasattr(memory, "is_pinned"):
            is_pinned = bool(getattr(memory, "is_pinned"))
        if not is_pinned and hasattr(memory, "metadata") and isinstance(memory.metadata, dict):
            is_pinned = bool(memory.metadata.get("is_pinned", False))

        if is_pinned or memory.importance_score >= 0.90:
            return "KEEP"

        # 2. Calculate decayed importance score
        current_score = self.scorer.calculate_decayed_importance(memory, now=current_time)
        if current_score < threshold:
            if memory.importance_score < 0.60:
                return "FORGET"
            elif memory.importance_score < 0.90:
                return "ARCHIVE"

        # 3. Check maximum inactivity days
        last_time = memory.last_accessed_at or memory.updated_at or memory.created_at
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        delta_days = (current_time - last_time).total_seconds() / 86400.0

        if max_inactivity_days and delta_days > max_inactivity_days:
            if memory.importance_score < 0.60:
                return "FORGET"
            elif memory.importance_score < 0.90:
                return "ARCHIVE"

        return "KEEP"

    def evaluate_memory_for_pruning(
        self,
        memory: MemoryItem,
        threshold: float = 0.20,
        max_inactivity_days: Optional[int] = 30,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Determine whether a memory item qualifies for forgetting/pruning.
        """
        return self.decide_lifecycle(
            memory=memory,
            threshold=threshold,
            max_inactivity_days=max_inactivity_days,
            now=now,
        ) == "FORGET"

    def prune_memories(self, request: PruneRequest) -> PruneResult:
        """
        Scan active memories and prune those that meet forgetting criteria.
        """
        all_candidates = self.metadata_store.list_memories(
            user_id=request.user_id,
            status=MemoryStatus.ACTIVE,
            limit=1000
        )

        pruned_ids: List[str] = []
        now = datetime.now(timezone.utc)

        for mem in all_candidates:
            should_prune = self.evaluate_memory_for_pruning(
                memory=mem,
                threshold=request.importance_threshold,
                max_inactivity_days=request.max_inactivity_days,
                now=now,
            )

            if should_prune:
                # 1. Remove from vector index
                try:
                    self.vector_store.delete_by_id(mem.id)
                except Exception as e:
                    logger.warning("Could not evict memory %s from vector store: %s", mem.id, str(e))

                if request.hard_delete:
                    # 2a. Hard delete from SQLite
                    self.metadata_store.delete_memory(mem.id)
                else:
                    # 2b. Soft delete: mark as FORGOTTEN in SQLite for dashboard inspection
                    self.metadata_store.update_status(mem.id, MemoryStatus.FORGOTTEN)

                pruned_ids.append(mem.id)

        # Log summary of pruning operation
        logger.info("Pruned %d memories out of %d candidates.", len(pruned_ids), len(all_candidates))

        return PruneResult(
            total_evaluated=len(all_candidates),
            pruned_count=len(pruned_ids),
            pruned_memory_ids=pruned_ids
        )
