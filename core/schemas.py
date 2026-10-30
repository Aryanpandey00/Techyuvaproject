"""
Pydantic v2 data transfer objects and validation schemas for AI Memory System.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, ConfigDict


class MemoryStatus(str, Enum):
    """Lifecycle status of a conversational memory item."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    COMPRESSED = "compressed"


class MemoryType(str, Enum):
    """Supported semantic memory categories."""
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    PERSONAL = "personal"
    TEMPORARY = "temporary"
    CONVERSATION_SUMMARY = "conversation_summary"


class LLMProviderType(str, Enum):
    """Supported LLM backend providers."""
    MOCK = "mock"
    OPENAI = "openai"
    LOCAL = "local"


class Memory(BaseModel):
    """
    Core Domain Model for a Memory item in the storage layer.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique memory identifier (UUID)")
    user_id: str = Field(default="default_user", description="User or session identifier")
    content: str = Field(..., min_length=1, description="Text content of the memory")
    memory_type: MemoryType = Field(default=MemoryType.FACT, description="Category of the memory")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score [0.0 - 1.0]")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score [0.0 - 1.0]")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")
    last_accessed_at: Optional[datetime] = Field(default=None, description="Last retrieval timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times this memory was retrieved")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="Lifecycle status")
    parent_ids: List[str] = Field(default_factory=list, description="IDs of source memories if compressed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary e.g. {'is_pinned': True}")

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @property
    def is_pinned(self) -> bool:
        """Helper property to check if memory is pinned against forgetting."""
        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            return bool(self.metadata.get("is_pinned", False))
        return False


# Alias for backward compatibility
MemoryItem = Memory


class ExtractedMemoryItem(BaseModel):
    """Structured extraction unit returned from LLM fact extraction."""
    content: str = Field(..., min_length=2, description="Atomic factual memory statement")
    memory_type: MemoryType = Field(default=MemoryType.FACT, description="Classified memory type")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated long-term importance [0.0 - 1.0]")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score [0.0 - 1.0]")

    model_config = ConfigDict(use_enum_values=True)


class MemoryCreate(BaseModel):
    """Payload for creating a new memory."""
    content: str = Field(..., min_length=1, description="Text content of the memory")
    user_id: str = Field(default="default_user", description="User identifier")
    memory_type: MemoryType = Field(default=MemoryType.FACT, description="Type of memory")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score [0.0 - 1.0]")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="Initial status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary e.g. {'is_pinned': True}")


class MemoryUpdate(BaseModel):
    """Payload for updating an existing memory."""
    content: Optional[str] = Field(default=None, min_length=1, description="Updated text content")
    user_id: Optional[str] = Field(default=None, description="Updated user ID")
    memory_type: Optional[MemoryType] = Field(default=None, description="Updated memory type")
    importance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Updated importance score")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Updated confidence score")
    status: Optional[MemoryStatus] = Field(default=None, description="Updated lifecycle status")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata dictionary")


class RelevanceClassification(str, Enum):
    """Relevance gate categorization for retrieved candidate memories."""
    RELEVANT = "relevant"
    WEAKLY_RELATED = "weakly_related"
    IRRELEVANT = "irrelevant"


class RetrievedMemory(BaseModel):
    """Memory item enriched with individual component scores and final re-ranked score."""
    memory: Memory
    similarity_score: float = Field(..., description="Vector cosine similarity [0.0 - 1.0]")
    importance_score: float = Field(default=0.5, description="Memory intrinsic importance [0.0 - 1.0]")
    recency_score: float = Field(default=1.0, description="Temporal recency decay score [0.0 - 1.0]")
    access_frequency_score: float = Field(default=0.0, description="Access frequency score [0.0 - 1.0]")
    final_score: float = Field(default=0.0, description="Weighted re-ranked score [0.0 - 1.0]")
    distance: float = Field(default=0.0, description="Vector cosine distance")
    relevance_status: str = Field(default="relevant", description="Relevance classification status")


# Backward compatibility aliases
MemorySearchResult = RetrievedMemory


class MemoryRetrieveRequest(BaseModel):
    """Query payload for semantic similarity search and re-ranking."""
    query: str = Field(..., min_length=1, description="Search query string")
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Alias for similarity_threshold")
    relevance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional relevance gate cutoff threshold")
    memory_type: Optional[MemoryType] = Field(default=None, description="Filter by memory type")
    status: Optional[MemoryStatus] = Field(default=MemoryStatus.ACTIVE, description="Filter by status (default active)")


class MemoryRetrieveResponse(BaseModel):
    """Response containing retrieved search results and prompt context."""
    query: str
    total_found: int
    retrieved_memories: List[RetrievedMemory]
    results: List[RetrievedMemory] = Field(default_factory=list, description="Alias for retrieved_memories")
    context_string: str = Field(default="", description="Formatted context string for LLM injection")


class MemoryStatsResponse(BaseModel):
    """Aggregate statistics for the memory database."""
    total_memories: int
    active_count: int
    archived_count: int
    forgotten_count: int
    compressed_count: int
    average_importance: float
    average_confidence: float
    vector_store_count: int


class SystemHealthResponse(BaseModel):
    """System health status and component availability."""
    status: str = Field(default="healthy")
    app_env: str
    llm_provider: str
    embedding_model: str
    chroma_status: str
    sqlite_status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    """Request for chat completion."""
    message: str = Field(..., min_length=1)
    user_id: str = Field(default="default_user")
    top_k_memories: int = Field(default=5, ge=1, le=20)
    auto_extract_memories: bool = Field(default=True)
    relevance_threshold: Optional[float] = Field(default=None, description="Optional override for relevance gate cutoff")
    max_context_tokens: Optional[int] = Field(default=None, description="Optional override for maximum injected context tokens")
    recent_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional compact recent turns for follow-up disambiguation without sending full unbounded history")


class ChatResponse(BaseModel):
    """Chat response with memory context and resource-efficiency metrics."""
    reply: str
    retrieved_memories: List[MemorySearchResult] = Field(default_factory=list)
    new_extracted_memories: List[Memory] = Field(default_factory=list)
    context_tokens_used: int = Field(default=0)
    injected_memories: List[MemorySearchResult] = Field(default_factory=list, description="Memories passing relevance gate and context budget")
    context_tokens_saved: int = Field(default=0, description="Estimated tokens saved by context optimization")
    context_reduction_percent: float = Field(default=0.0, description="Percentage reduction in context size")
    retrieval_latency_ms: float = Field(default=0.0, description="Latency in milliseconds for semantic retrieval and relevance gating")
    memory_used: bool = Field(default=False, description="Whether any long-term memory context was injected into the prompt")
    relevance_status: str = Field(default="none", description="Relevance classification status")


class ResourceEfficiencyMetrics(BaseModel):
    """Real-time and benchmark resource efficiency metrics."""
    original_context_chars: int = Field(default=0)
    compressed_context_chars: int = Field(default=0)
    estimated_input_tokens: int = Field(default=0)
    estimated_tokens_saved: int = Field(default=0)
    percentage_context_reduction: float = Field(default=0.0)
    candidate_memories_retrieved: int = Field(default=0)
    memories_injected: int = Field(default=0)
    memories_filtered: int = Field(default=0)
    retrieval_latency_ms: float = Field(default=0.0)
    compression_ratio: float = Field(default=1.0)


class CompressionStats(BaseModel):
    """Statistics and quantitative metrics of a memory compression operation."""
    original_memory_count: int = Field(..., description="Number of original memories merged")
    original_char_count: int = Field(..., description="Total characters across original memories")
    compressed_char_count: int = Field(..., description="Characters in synthesized compressed memory")
    compressed_memory_size: int = Field(..., description="Size of compressed memory (chars)")
    compression_ratio: float = Field(..., description="Ratio of compressed size to original size (e.g. 0.45)")
    information_retained: float = Field(default=0.95, description="Estimated information retention percentage [0.0 - 1.0]")
    tokens_saved_est: int = Field(default=0, description="Estimated tokens saved for future prompts")


class CompressionRequest(BaseModel):
    """Request payload to trigger semantic clustering and compression of memories."""
    memory_ids: Optional[List[str]] = Field(default=None, description="Explicit list of memory IDs to compress")
    user_id: Optional[str] = Field(default=None, description="Filter memories by user ID")
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Cosine similarity cutoff for clustering")
    min_cluster_size: int = Field(default=2, ge=2, description="Minimum memories required to form a cluster")
    max_memory_length: int = Field(default=250, ge=20, le=1000, description="Maximum character length of synthesized memory")
    dry_run: bool = Field(default=False, description="If True, returns preview without mutating database")


class CompressionResult(BaseModel):
    """Result summary of a memory compression run."""
    compressed_memory: Optional[Memory] = None
    original_memory_ids: List[str] = Field(default_factory=list)
    stats: Optional[CompressionStats] = None
    clusters_found: int = 0
    memories_compressed_count: int = 0
    new_summaries_created: int = 0
    compressed_items: List[Memory] = Field(default_factory=list)


class PruneRequest(BaseModel):
    """Placeholder prune request for forward compatibility."""
    user_id: Optional[str] = None
    importance_threshold: float = 0.20
    max_inactivity_days: Optional[int] = 30
    hard_delete: bool = False


class PruneResult(BaseModel):
    """Placeholder prune result for forward compatibility."""
    total_evaluated: int = 0
    pruned_count: int = 0
    pruned_memory_ids: List[str] = Field(default_factory=list)
