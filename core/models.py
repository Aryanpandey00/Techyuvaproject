"""
Internal data models and SQLite database table definitions for Memory items.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import List, Optional
from core.schemas import Memory, MemoryStatus, MemoryType


CREATE_MEMORIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default_user',
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance_score REAL NOT NULL DEFAULT 0.5,
    confidence_score REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    parent_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
"""


@dataclass
class MemoryRecord:
    """Dataclass representing a raw SQLite memory row."""
    id: str
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    confidence_score: float
    created_at: str
    updated_at: str
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    status: str = "active"
    parent_ids: str = "[]"
    metadata: str = "{}"

    def to_schema(self) -> Memory:
        """Convert SQLite row dataclass to Pydantic Memory schema."""
        parsed_parent_ids = []
        try:
            if self.parent_ids:
                parsed = json.loads(self.parent_ids)
                if isinstance(parsed, list):
                    parsed_parent_ids = [str(x) for x in parsed]
        except Exception:
            parsed_parent_ids = []

        parsed_metadata = {}
        try:
            meta_str = getattr(self, "metadata", None)
            if meta_str:
                parsed = json.loads(meta_str)
                if isinstance(parsed, dict):
                    parsed_metadata = parsed
        except Exception:
            parsed_metadata = {}

        return Memory(
            id=self.id,
            user_id=self.user_id,
            content=self.content,
            memory_type=MemoryType(self.memory_type),
            importance_score=float(self.importance_score),
            confidence_score=float(self.confidence_score),
            created_at=datetime.fromisoformat(self.created_at),
            updated_at=datetime.fromisoformat(self.updated_at),
            last_accessed_at=datetime.fromisoformat(self.last_accessed_at) if self.last_accessed_at else None,
            access_count=int(self.access_count),
            status=MemoryStatus(self.status),
            parent_ids=parsed_parent_ids,
            metadata=parsed_metadata,
        )

    @classmethod
    def from_schema(cls, item: Memory) -> "MemoryRecord":
        """Convert Pydantic Memory schema to SQLite row dataclass."""
        return cls(
            id=item.id,
            user_id=item.user_id,
            content=item.content,
            memory_type=item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
            importance_score=item.importance_score,
            confidence_score=item.confidence_score,
            created_at=item.created_at.isoformat(),
            updated_at=item.updated_at.isoformat(),
            last_accessed_at=item.last_accessed_at.isoformat() if item.last_accessed_at else None,
            access_count=item.access_count,
            status=item.status.value if hasattr(item.status, "value") else str(item.status),
            parent_ids=json.dumps(getattr(item, "parent_ids", []) or []),
            metadata=json.dumps(getattr(item, "metadata", {}) or {}),
        )
