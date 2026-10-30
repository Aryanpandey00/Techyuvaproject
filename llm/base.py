"""
Abstract Base Class for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM operations: conversational response generation,
    fact extraction, memory synthesis/compression, and importance scoring.
    """

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a contextual assistant response given the user prompt, optional memories, and compact history."""
        pass

    @abstractmethod
    def extract_memories(
        self,
        user_message: str,
        assistant_reply: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract factual statements, user preferences, or semantic memory units
        from a conversational turn.
        Returns: List of dicts with keys: 'content', 'memory_type', 'importance_score'
        """
        pass

    @abstractmethod
    def compress_memories(
        self,
        memory_contents: List[str],
    ) -> str:
        """
        Synthesize a cluster of related memories into a single concise,
        information-dense memory statement.
        """
        pass

    @abstractmethod
    def score_importance(
        self,
        memory_content: str,
    ) -> float:
        """
        Evaluate and assign an intrinsic importance score [0.0 - 1.0]
        to a piece of information.
        """
        pass
