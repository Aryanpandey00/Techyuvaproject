"""
Custom domain exceptions for AI Memory Compression System.
"""
from typing import Optional, Any, Dict


class MemorySystemException(Exception):
    """Base exception for all domain-specific errors in the memory system."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MemoryNotFoundError(MemorySystemException):
    """Raised when a requested memory ID does not exist in vector or metadata storage."""
    pass


class StorageError(MemorySystemException):
    """Raised when an operation on ChromaDB or SQLite fails."""
    pass


class EmbeddingError(MemorySystemException):
    """Raised when embedding generation or dimension validation fails."""
    pass


class CompressionError(MemorySystemException):
    """Raised when memory clustering or LLM synthesis fails during compression."""
    pass


class LLMProviderError(MemorySystemException):
    """Raised when an LLM API call, timeout, or parsing error occurs."""
    pass


class ForgettingPolicyError(MemorySystemException):
    """Raised when memory pruning or status transition violates lifecycle rules."""
    pass
