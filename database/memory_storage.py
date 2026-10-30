"""
Unified Memory Storage Service orchestrating SQLite metadata and ChromaDB vector persistence.
"""
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional
from core.exceptions import MemoryNotFoundError, StorageError
from core.schemas import (
    Memory,
    MemoryCreate,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    MemoryStatsResponse,
)
from database.chroma_store import ChromaVectorStore
from database.sqlite_store import SQLiteMetadataStore
from embeddings.encoder import BaseEmbeddingEncoder, get_embedding_encoder

logger = logging.getLogger(__name__)


class MemoryStorageService:
    """
    Unified storage service managing both relational metadata (SQLite)
    and high-dimensional vector embeddings (ChromaDB).
    """

    def __init__(
        self,
        sqlite_db_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        chroma_collection_name: Optional[str] = None,
        encoder: Optional[BaseEmbeddingEncoder] = None,
        in_memory: bool = False,
    ) -> None:
        self.sqlite_path = sqlite_db_path or os.getenv("SQLITE_DB_PATH", "./data/memory_metadata.db")
        self.chroma_dir = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.chroma_collection = chroma_collection_name or os.getenv("CHROMA_COLLECTION_NAME", "llm_conversational_memories")
        self.in_memory = in_memory

        # 1. Initialize SQLite Store
        self.sqlite_store = SQLiteMetadataStore(
            db_path=":memory:" if self.in_memory else self.sqlite_path
        )

        # 2. Initialize ChromaDB Vector Store
        self.chroma_store = ChromaVectorStore(
            persist_directory=None if self.in_memory else self.chroma_dir,
            collection_name=self.chroma_collection,
            in_memory=self.in_memory,
        )

        # 3. Initialize Embedding Encoder
        self.encoder = encoder or get_embedding_encoder()
        logger.info("MemoryStorageService initialized successfully.")

    def create_memory(self, item: MemoryCreate) -> Memory:
        """
        Create a new memory:
        1. Instantiate Memory domain model
        2. Generate dense vector embedding via Sentence Transformer
        3. Save metadata to SQLite
        4. Upsert vector to ChromaDB
        """
        memory = Memory(
            content=item.content,
            user_id=item.user_id,
            memory_type=item.memory_type,
            importance_score=item.importance_score,
            confidence_score=item.confidence_score,
            status=item.status,
            metadata=item.metadata,
        )

        # Generate embedding
        embedding = self.encoder.encode_text(memory.content)

        # Persist in SQLite
        self.sqlite_store.save_memory(memory)

        # Persist in ChromaDB
        status_val = memory.status.value if hasattr(memory.status, "value") else str(memory.status)
        type_val = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)

        self.chroma_store.upsert_embedding(
            memory_id=memory.id,
            embedding=embedding,
            document=memory.content,
            metadata={
                "user_id": memory.user_id,
                "memory_type": type_val,
                "status": status_val,
                "importance_score": memory.importance_score,
                "confidence_score": memory.confidence_score,
                "is_pinned": bool(memory.is_pinned),
            }
        )

        logger.info("Created memory: %s (user: %s, type: %s)", memory.id, memory.user_id, type_val)
        return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by unique ID from SQLite metadata store."""
        return self.sqlite_store.get_memory(memory_id)

    def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory:
        """
        Update an existing memory:
        - If content changed, re-encodes and updates ChromaDB vector index.
        - Updates metadata in SQLite and ChromaDB.
        - Consistently evicts or re-indexes vector when status transitions.
        """
        updated_memory = self.sqlite_store.update_memory(memory_id, update)

        # If marked FORGOTTEN, evict from vector store immediately
        if updated_memory.status == MemoryStatus.FORGOTTEN:
            self.chroma_store.delete_by_id(memory_id)
        elif update.content is not None or update.status is not None or update.memory_type is not None or update.metadata is not None:
            # Re-index in ChromaDB
            embedding = self.encoder.encode_text(updated_memory.content)
            status_val = updated_memory.status.value if hasattr(updated_memory.status, "value") else str(updated_memory.status)
            type_val = updated_memory.memory_type.value if hasattr(updated_memory.memory_type, "value") else str(updated_memory.memory_type)
            self.chroma_store.upsert_embedding(
                memory_id=updated_memory.id,
                embedding=embedding,
                document=updated_memory.content,
                metadata={
                    "user_id": updated_memory.user_id,
                    "memory_type": type_val,
                    "status": status_val,
                    "importance_score": updated_memory.importance_score,
                    "confidence_score": updated_memory.confidence_score,
                    "is_pinned": bool(updated_memory.is_pinned),
                }
            )

        logger.info("Updated memory: %s", memory_id)
        return updated_memory

    def update_status(self, memory_id: str, new_status: MemoryStatus) -> Memory:
        """Update memory lifecycle status (e.g. active, archived, forgotten)."""
        return self.update_memory(memory_id, MemoryUpdate(status=new_status))

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete memory from both ChromaDB vector store and SQLite metadata store."""
        try:
            self.chroma_store.delete_by_id(memory_id)
        except Exception as e:
            logger.warning("Could not delete from Chroma vector store: %s", str(e))

        return self.sqlite_store.delete_memory(memory_id)

    def list_memories(
        self,
        user_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[float] = None,
        min_confidence: Optional[float] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        """List memories matching specified filter parameters."""
        return self.sqlite_store.list_memories(
            user_id=user_id,
            status=status,
            memory_type=memory_type,
            min_importance=min_importance,
            min_confidence=min_confidence,
            search_query=search_query,
            limit=limit,
            offset=offset,
        )

    def search_similar_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
    ) -> List[MemorySearchResult]:
        """
        Semantic similarity search:
        1. Encodes query text with Sentence Transformer
        2. Queries ChromaDB with optional metadata filters (user_id, status, memory_type)
        3. Fetches full memory records from SQLite
        4. Records access timestamp/count
        5. Returns ranked list of MemorySearchResult
        """
        if not query or not query.strip():
            return []

        query_embedding = self.encoder.encode_text(query)

        # Build Chroma where filter
        where_filter: Dict[str, Any] = {}
        if user_id:
            where_filter["user_id"] = user_id
        if status:
            where_filter["status"] = status.value if hasattr(status, "value") else str(status)
        if memory_type:
            where_filter["memory_type"] = memory_type.value if hasattr(memory_type, "value") else str(memory_type)

        raw_results = self.chroma_store.query_similarity(
            query_embedding=query_embedding,
            top_k=top_k,
            where_filter=where_filter if where_filter else None,
        )

        search_results: List[MemorySearchResult] = []
        for mem_id, similarity, _, _ in raw_results:
            if similarity < min_similarity:
                continue

            memory = self.sqlite_store.get_memory(mem_id)
            if not memory:
                continue

            # Ensure forgotten memories are excluded unless explicitly queried
            if memory.status == MemoryStatus.FORGOTTEN and status != MemoryStatus.FORGOTTEN:
                continue

            # Record access in SQLite
            self.sqlite_store.record_access(mem_id)

            search_results.append(MemorySearchResult(
                memory=memory,
                similarity_score=round(similarity, 4),
                distance=round(1.0 - similarity, 4),
            ))

        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results

    def get_stats(self, user_id: Optional[str] = None) -> MemoryStatsResponse:
        """Aggregate counts and diagnostics across SQLite and ChromaDB."""
        sqlite_stats = self.sqlite_store.get_stats(user_id=user_id)
        chroma_count = self.chroma_store.count()

        return MemoryStatsResponse(
            total_memories=sqlite_stats["total"],
            active_count=sqlite_stats["active"],
            archived_count=sqlite_stats["archived"],
            forgotten_count=sqlite_stats["forgotten"],
            compressed_count=sqlite_stats["compressed"],
            average_importance=sqlite_stats["avg_importance"],
            average_confidence=sqlite_stats["avg_confidence"],
            vector_store_count=chroma_count,
        )
