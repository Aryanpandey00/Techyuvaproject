"""
SQLite Metadata Store for Memory storage, lifecycle, and access auditing.
"""
from datetime import datetime, timezone
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional
from core.exceptions import StorageError, MemoryNotFoundError
from core.models import CREATE_MEMORIES_TABLE_SQL, MemoryRecord
from core.schemas import Memory, MemoryStatus, MemoryType, MemoryUpdate

logger = logging.getLogger(__name__)


class SQLiteMetadataStore:
    """Provides relational ACID storage for Memory items and metadata."""

    def __init__(self, db_path: str = "./data/memory_metadata.db") -> None:
        self.db_path = db_path
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create or return an SQLite database connection with row factory enabled."""
        if self._mem_conn is not None:
            return self._mem_conn
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        """Execute table creation schema and indexes."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.executescript(CREATE_MEMORIES_TABLE_SQL)
            # Auto-migrate parent_ids column if missing
            try:
                cursor.execute("ALTER TABLE memories ADD COLUMN parent_ids TEXT NOT NULL DEFAULT '[]'")
            except Exception:
                pass
            # Auto-migrate metadata column if missing
            try:
                cursor.execute("ALTER TABLE memories ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            except Exception:
                pass
            conn.commit()
            if self._mem_conn is None:
                conn.close()
            logger.info("SQLiteMetadataStore initialized at: %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize SQLite database: %s", str(e))
            raise StorageError(f"SQLite initialization failed: {str(e)}") from e

    def save_memory(self, memory: Memory) -> None:
        """Insert or replace a memory record in the database."""
        record = MemoryRecord.from_schema(memory)
        sql = """
        INSERT OR REPLACE INTO memories (
            id, user_id, content, memory_type, importance_score, confidence_score,
            created_at, updated_at, last_accessed_at, access_count, status, parent_ids, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    record.id,
                    record.user_id,
                    record.content,
                    record.memory_type,
                    record.importance_score,
                    record.confidence_score,
                    record.created_at,
                    record.updated_at,
                    record.last_accessed_at,
                    record.access_count,
                    record.status,
                    record.parent_ids,
                    record.metadata,
                ))
                conn.commit()
        except Exception as e:
            logger.error("Failed to save memory %s: %s", memory.id, str(e))
            raise StorageError(f"Failed to save memory metadata: {str(e)}") from e

    def save_batch(self, memories: List[Memory]) -> None:
        """Batch insert or replace memories."""
        if not memories:
            return
        sql = """
        INSERT OR REPLACE INTO memories (
            id, user_id, content, memory_type, importance_score, confidence_score,
            created_at, updated_at, last_accessed_at, access_count, status, parent_ids, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                r.id, r.user_id, r.content, r.memory_type, r.importance_score,
                r.confidence_score, r.created_at, r.updated_at, r.last_accessed_at,
                r.access_count, r.status, r.parent_ids, r.metadata
            )
            for r in (MemoryRecord.from_schema(m) for m in memories)
        ]
        try:
            with self._get_connection() as conn:
                conn.executemany(sql, params)
                conn.commit()
        except Exception as e:
            logger.error("Failed to batch save memories: %s", str(e))
            raise StorageError(f"Batch metadata save failed: {str(e)}") from e

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single memory by ID."""
        sql = "SELECT * FROM memories WHERE id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (memory_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return MemoryRecord(**dict(row)).to_schema()
        except Exception as e:
            logger.error("Failed to fetch memory %s: %s", memory_id, str(e))
            raise StorageError(f"Error reading memory: {str(e)}") from e

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
        conditions = []
        params: List[Any] = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            status_val = status.value if hasattr(status, "value") else str(status)
            conditions.append("status = ?")
            params.append(status_val)
        if memory_type is not None:
            type_val = memory_type.value if hasattr(memory_type, "value") else str(memory_type)
            conditions.append("memory_type = ?")
            params.append(type_val)
        if min_importance is not None:
            conditions.append("importance_score >= ?")
            params.append(min_importance)
        if min_confidence is not None:
            conditions.append("confidence_score >= ?")
            params.append(min_confidence)
        if search_query:
            conditions.append("content LIKE ?")
            params.append(f"%{search_query}%")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM memories {where_clause} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                return [MemoryRecord(**dict(row)).to_schema() for row in rows]
        except Exception as e:
            logger.error("Failed to query memories list: %s", str(e))
            raise StorageError(f"Failed to query memories: {str(e)}") from e

    def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory:
        """Update memory fields selectively."""
        current = self.get_memory(memory_id)
        if not current:
            raise MemoryNotFoundError(f"Memory with id '{memory_id}' not found")

        updated_dict = current.model_dump()
        for field_name, value in update.model_dump(exclude_unset=True).items():
            if value is not None:
                updated_dict[field_name] = value

        updated_dict["updated_at"] = datetime.now(timezone.utc)
        updated_item = Memory(**updated_dict)
        self.save_memory(updated_item)
        return updated_item

    def update_status(self, memory_id: str, new_status: MemoryStatus) -> Memory:
        """Update memory lifecycle status."""
        current = self.get_memory(memory_id)
        if not current:
            raise MemoryNotFoundError(f"Memory with id '{memory_id}' not found")

        status_val = new_status.value if hasattr(new_status, "value") else str(new_status)
        now = datetime.now(timezone.utc).isoformat()
        sql = "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?"
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (status_val, now, memory_id))
                conn.commit()
            return self.get_memory(memory_id)
        except Exception as e:
            logger.error("Failed to update status for memory %s: %s", memory_id, str(e))
            raise StorageError(f"Status update failed: {str(e)}") from e

    def record_access(self, memory_id: str) -> None:
        """Increment access count and timestamp when memory is retrieved."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        UPDATE memories
        SET access_count = access_count + 1,
            last_accessed_at = ?,
            updated_at = ?
        WHERE id = ?
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (now, now, memory_id))
                conn.commit()
        except Exception as e:
            logger.error("Failed to record access for memory %s: %s", memory_id, str(e))

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Failed to delete memory %s: %s", memory_id, str(e))
            raise StorageError(f"Delete failed: {str(e)}") from e

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate counts and stats from SQLite database."""
        conditions = []
        params = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
            SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) as archived,
            SUM(CASE WHEN status = 'forgotten' THEN 1 ELSE 0 END) as forgotten,
            SUM(CASE WHEN status = 'compressed' THEN 1 ELSE 0 END) as compressed,
            AVG(importance_score) as avg_importance,
            AVG(confidence_score) as avg_confidence
        FROM memories {where_clause}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, tuple(params))
                row = cursor.fetchone()
                if not row:
                    return {
                        "total": 0, "active": 0, "archived": 0, "forgotten": 0,
                        "compressed": 0, "avg_importance": 0.0, "avg_confidence": 0.0
                    }
                return {
                    "total": row["total"] or 0,
                    "active": row["active"] or 0,
                    "archived": row["archived"] or 0,
                    "forgotten": row["forgotten"] or 0,
                    "compressed": row["compressed"] or 0,
                    "avg_importance": round(row["avg_importance"] or 0.0, 3),
                    "avg_confidence": round(row["avg_confidence"] or 0.0, 3),
                }
        except Exception as e:
            logger.error("Failed to compute stats: %s", str(e))
            return {
                "total": 0, "active": 0, "archived": 0, "forgotten": 0,
                "compressed": 0, "avg_importance": 0.0, "avg_confidence": 0.0
            }
