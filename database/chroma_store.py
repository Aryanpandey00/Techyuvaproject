"""
ChromaDB Vector Store implementation with resilient fallback for semantic vector indexing and similarity search.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import numpy as np
from core.exceptions import StorageError

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Manages high-dimensional vector embeddings and similarity queries via ChromaDB,
    with an in-memory vector index fallback.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = "./data/chroma_db",
        collection_name: str = "llm_conversational_memories",
        in_memory: bool = False,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.in_memory = in_memory
        self._client = None
        self._collection = None
        self._fallback_mode = False
        # In-memory dictionary fallback store: {id: {"embedding": [...], "document": str, "metadata": dict}}
        self._fallback_vectors: Dict[str, Dict[str, Any]] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client with persistent local storage or in-memory fallback."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            if self.in_memory or not self.persist_directory:
                self._client = chromadb.Client(ChromaSettings(anonymized_telemetry=False, is_persistent=False))
            else:
                os.makedirs(os.path.abspath(self.persist_directory), exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._fallback_mode = False
            logger.info("ChromaVectorStore initialized via ChromaDB. Collection: %s", self.collection_name)
        except Exception as e:
            logger.warning("ChromaDB native client not available (%s). Using high-performance in-memory vector store.", str(e))
            self._fallback_mode = True

    def upsert_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert a single memory embedding, text content, and metadata."""
        sanitized_meta = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in (metadata or {}).items()
        }

        if self._fallback_mode or not self._collection:
            self._fallback_vectors[memory_id] = {
                "embedding": np.array(embedding, dtype=np.float32),
                "document": document,
                "metadata": sanitized_meta,
            }
            return

        try:
            self._collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[sanitized_meta]
            )
        except Exception as e:
            logger.error("Error upserting embedding for memory %s: %s", memory_id, str(e))
            raise StorageError(f"Vector upsert failed: {str(e)}") from e

    def upsert_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch upsert multiple memory vectors."""
        if not ids:
            return

        for idx, mem_id in enumerate(ids):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            self.upsert_embedding(
                memory_id=mem_id,
                embedding=embeddings[idx],
                document=documents[idx],
                metadata=meta
            )

    def query_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """
        Query the nearest memories by vector cosine similarity.
        Returns: List of tuples (memory_id, similarity_score, document_text, metadata)
        """
        if self._fallback_mode or not self._collection:
            if not self._fallback_vectors:
                return []

            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            scored_items = []
            for mem_id, item in self._fallback_vectors.items():
                meta = item["metadata"]
                # Apply where filter
                if where_filter:
                    match = True
                    for k, v in where_filter.items():
                        if k == "$and" and isinstance(v, list):
                            for sub in v:
                                for sub_k, sub_v in sub.items():
                                    if meta.get(sub_k) != sub_v:
                                        match = False
                                        break
                        elif meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                t_vec = item["embedding"]
                t_norm = np.linalg.norm(t_vec)
                if t_norm > 0:
                    t_vec = t_vec / t_norm

                sim = float(np.dot(q_vec, t_vec))
                similarity = max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))
                scored_items.append((mem_id, similarity, item["document"], meta))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            return scored_items[:top_k]

        try:
            count = self.count()
            if count == 0:
                return []
            
            k = min(top_k, count)
            sanitized_filter = None
            if where_filter:
                conditions = []
                for key, val in where_filter.items():
                    if val is not None:
                        conditions.append({key: val})
                if len(conditions) == 1:
                    sanitized_filter = conditions[0]
                elif len(conditions) > 1:
                    sanitized_filter = {"$and": conditions}

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=sanitized_filter,
                include=["documents", "metadatas", "distances"]
            )

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            retrieved = []
            for mem_id, doc, meta, dist in zip(ids, docs, metas, distances):
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                retrieved.append((mem_id, similarity, doc, meta or {}))

            return retrieved
        except Exception as e:
            logger.error("Vector similarity search failed: %s", str(e))
            raise StorageError(f"Vector search failed: {str(e)}") from e

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored vector metadata by ID."""
        if self._fallback_mode or not self._collection:
            item = self._fallback_vectors.get(memory_id)
            if not item:
                return None
            return {
                "id": memory_id,
                "document": item["document"],
                "metadata": item["metadata"],
            }

        try:
            res = self._collection.get(ids=[memory_id], include=["documents", "metadatas"])
            ids = res.get("ids", [])
            if not ids:
                return None
            return {
                "id": ids[0],
                "document": res.get("documents", [""])[0],
                "metadata": res.get("metadatas", [{}])[0],
            }
        except Exception as e:
            logger.error("Failed to get vector %s: %s", memory_id, str(e))
            return None

    def delete_by_id(self, memory_id: str) -> None:
        """Delete a vector entry by memory ID."""
        if self._fallback_mode or not self._collection:
            self._fallback_vectors.pop(memory_id, None)
            return

        try:
            self._collection.delete(ids=[memory_id])
        except Exception as e:
            logger.error("Failed to delete vector %s: %s", memory_id, str(e))
            raise StorageError(f"Vector deletion failed: {str(e)}") from e

    def delete_batch(self, memory_ids: List[str]) -> None:
        """Delete multiple vectors by IDs."""
        for m_id in memory_ids:
            self.delete_by_id(m_id)

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        if self._fallback_mode or not self._collection:
            return len(self._fallback_vectors)

        try:
            return self._collection.count()
        except Exception as e:
            logger.error("Error counting Chroma vectors: %s", str(e))
            return 0

    def reset_collection(self) -> None:
        """Reset/clear all items in collection."""
        if self._fallback_mode or not self._collection:
            self._fallback_vectors.clear()
            return

        try:
            if self._client and self._collection:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.error("Failed to reset Chroma collection: %s", str(e))
            raise StorageError(f"Collection reset failed: {str(e)}") from e
