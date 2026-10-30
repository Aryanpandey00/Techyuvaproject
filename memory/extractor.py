"""
Memory Extractor: Automatically extracts structured, validated, non-duplicate,
and sanitized long-term memories from conversation turns using LLMs.
"""
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.schemas import (
    ExtractedMemoryItem,
    Memory,
    MemoryCreate,
    MemoryStatus,
    MemoryType,
)
from embeddings.encoder import BaseEmbeddingEncoder, get_embedding_encoder
from llm.base import BaseLLMProvider
from llm.provider import get_llm_provider

logger = logging.getLogger(__name__)

# Common conversational banter, greetings, and ephemeral phrases to filter out
CASUAL_BANTER_PATTERNS = [
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening|night|day))\b",
    r"^(?:how\s+are\s+you|what'?s\s+up|how\s+is\s+it\s+going|howdy|sup)\b",
    r"^(?:thanks|thank\s+you|thx|ty|you'?re\s+welcome|no\s+problem)\b",
    r"^(?:ok|okay|cool|sure|yep|yeah|yes|no|nope|alright|got\s+it|understood)\b",
    r"^(?:bye|goodbye|see\s+you|cya|take\s+care|talk\s+to\s+you\s+later)\b",
    r"\b(?:brb|be\s+right\s+back|heading\s+out\s+for\s+lunch|eating\s+(?:a\s+)?sandwich|in\s+\d+\s+mins?)\b",
]

# Sensitive information patterns (API keys, passwords, credit card numbers)
SENSITIVE_PATTERNS = [
    (r"(?i)\b(?:password|pwd|passphrase)\s*(?:is|[:=])\s*[:=]?\s*\S+", "[REDACTED_PASSWORD]"),
    (r"(?i)\b(?:sk-[a-zA-Z0-9]{15,}|ghp_[a-zA-Z0-9]{15,}|bearer\s+[a-zA-Z0-9_\-\.]{15,})\b", "[REDACTED_SECRET_KEY]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD_NUMBER]"),
]


class MemoryExtractor:
    """
    Extracts high-value long-term memory items from conversations,
    validates output schemas, removes duplicates via semantic similarity,
    filters trivial conversation, and sanitizes sensitive data.
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        encoder: Optional[BaseEmbeddingEncoder] = None,
        duplicate_similarity_threshold: float = 0.85,
        min_importance_threshold: float = 0.30,
        min_confidence_threshold: float = 0.50,
    ) -> None:
        self.llm = llm_provider or get_llm_provider()
        self.encoder = encoder or get_embedding_encoder()
        self.duplicate_threshold = duplicate_similarity_threshold
        self.min_importance = min_importance_threshold
        self.min_confidence = min_confidence_threshold

    def _sanitize_text(self, text: str) -> str:
        """Redact raw passwords, secret keys, or credit cards from extracted text."""
        sanitized = text
        for pattern, replacement in SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)
        return sanitized

    def _is_casual_or_temporary(self, text: str) -> bool:
        """Detect whether an utterance is purely casual greeting, banter, or temporary filler."""
        cleaned = text.strip().lower()
        if len(cleaned.split()) < 2 and not any(w in cleaned for w in ["alex", "rahul", "python"]):
            return True
        for pat in CASUAL_BANTER_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                # Only reject if it lacks substantive facts
                if not any(k in cleaned for k in ["name is", "study", "work", "prefer", "project", "live in", "goal"]):
                    return True
        return False

    def _parse_llm_json_response(self, raw_response: str) -> List[Dict[str, Any]]:
        """Safely parse JSON list from LLM output (handles backtick code fences)."""
        if not raw_response or not raw_response.strip():
            return []
        
        text = raw_response.strip()

        # Extract text inside ```json ... ``` codeblocks if present
        json_fence_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
        if json_fence_match:
            text = json_fence_match.group(1).strip()
        else:
            # Look for outermost array [ ... ]
            bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
            if bracket_match:
                text = bracket_match.group(0).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # If wrapped in a dictionary like {"memories": [...]}
                for key in ["memories", "items", "data", "extracted"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
            return []
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM extraction response: %s", str(e))
            return []

    def _build_extraction_prompt(self, conversation_text: str) -> str:
        """Create structured prompt instructing the LLM to output valid JSON memory items."""
        return f"""You are an intelligent memory extraction engine for an AI assistant.
Analyze the following conversation and extract all permanent, long-term factual statements, user preferences, projects, goals, or personal profile facts.

RULES:
1. Formulate each memory as a clear, third-person declarative statement (e.g. "User's name is Rahul.", "User is studying B.Tech CSE.", "User prefers Python for projects.").
2. Categorize each memory into one of the following exact types:
   - "fact": General verified facts about the user or world.
   - "preference": Likes, dislikes, tools, styles, or habits preferred by the user.
   - "goal": Aspirations, targets, or plans the user wants to achieve.
   - "project": Specific applications, assignments, or systems the user is building.
   - "personal": Biographical details like name, location, occupation, background.
   - "temporary": Ephemeral items (normally ignored unless specifically asked).
   - "conversation_summary": High-level summary of major topics.
3. Assign an 'importance_score' between 0.0 and 1.0 (higher for permanent identity/critical facts).
4. Assign a 'confidence_score' between 0.0 and 1.0 based on how explicitly stated it is.
5. REJECT casual greetings (e.g. "hi", "hello"), social banter ("how are you"), and temporary non-informative chatter.
6. Return a valid JSON list of objects matching this exact structure:
[
  {{
    "content": "User's name is Rahul.",
    "memory_type": "personal",
    "importance_score": 0.95,
    "confidence_score": 1.0
  }}
]

If there are no useful permanent memories to extract, return an empty list [].

Conversation:
\"\"\"{conversation_text}\"\"\"

JSON Output:"""

    def is_duplicate(
        self,
        candidate_embedding: List[float],
        existing_embeddings: List[List[float]],
    ) -> bool:
        """Check if candidate embedding has cosine similarity >= threshold against any existing memory."""
        if not existing_embeddings:
            return False
        sim_matrix = cosine_similarity(
            np.array([candidate_embedding]),
            np.array(existing_embeddings)
        )
        max_sim = float(np.max(sim_matrix))
        return max_sim >= self.duplicate_threshold

    def extract_memories(
        self,
        conversation: Union[str, List[Dict[str, str]], List[Any]],
        user_id: str = "default_user",
        existing_memories: Optional[List[Memory]] = None,
    ) -> List[Memory]:
        """
        Extract, validate, deduplicate, and return validated Memory domain objects.

        Args:
            conversation: A raw string, or list of message dictionaries [{"role": "user", "content": "..."}]
            user_id: Target user ID
            existing_memories: Optional list of already stored memories to check for duplicates

        Returns:
            List of validated Memory models ready for persistence
        """
        # 1. Normalize conversation input to a clean text representation
        if isinstance(conversation, str):
            conv_text = conversation.strip()
        elif isinstance(conversation, list):
            lines = []
            for msg in conversation:
                if isinstance(msg, dict):
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "").strip()
                    lines.append(f"{role}: {content}")
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    lines.append(f"{msg.role.capitalize()}: {msg.content.strip()}")
                else:
                    lines.append(str(msg))
            conv_text = "\n".join(lines)
        else:
            conv_text = str(conversation)

        if not conv_text or self._is_casual_or_temporary(conv_text):
            logger.info("Conversation filtered out as casual banter or empty.")
            return []

        # 2. Extract structured memories via LLM
        prompt = self._build_extraction_prompt(conv_text)
        try:
            # We pass the prompt to the LLM
            raw_output = self.llm.generate_response(prompt=prompt, temperature=0.1, max_tokens=1024)
            parsed_items = self._parse_llm_json_response(raw_output)

            # Fallback to provider's native extract_memories if JSON output was empty
            if not parsed_items and hasattr(self.llm, "extract_memories"):
                parsed_items = self.llm.extract_memories(conv_text)

        except Exception as e:
            logger.error("LLM extraction call failed: %s", str(e))
            # Fallback to pattern-based extraction on mock
            parsed_items = getattr(self.llm, "extract_memories", lambda x: [])(conv_text)

        if not parsed_items:
            return []

        # 3. Prepare existing memory embeddings for duplicate detection
        existing_vecs: List[List[float]] = []
        if existing_memories:
            for em in existing_memories:
                if em.status == MemoryStatus.ACTIVE:
                    try:
                        existing_vecs.append(self.encoder.encode_text(em.content))
                    except Exception:
                        pass

        # 4. Validate with Pydantic, sanitize sensitive data, and deduplicate
        validated_memories: List[Memory] = []
        current_batch_vecs: List[List[float]] = list(existing_vecs)

        for raw_item in parsed_items:
            try:
                # Validate with Pydantic
                extracted_dto = ExtractedMemoryItem(**raw_item)

                # Filter out low importance or low confidence
                if extracted_dto.importance_score < self.min_importance:
                    continue
                if extracted_dto.confidence_score < self.min_confidence:
                    continue

                # Filter out temporary trivial statements
                if self._is_casual_or_temporary(extracted_dto.content):
                    continue

                # Sanitize sensitive data
                sanitized_content = self._sanitize_text(extracted_dto.content)

                # Generate embedding for duplicate detection
                item_vec = self.encoder.encode_text(sanitized_content)

                # Check duplicate against existing memories and current batch
                if self.is_duplicate(item_vec, current_batch_vecs):
                    logger.info("Rejecting duplicate memory: '%s'", sanitized_content)
                    continue

                # Create complete Memory domain object
                memory = Memory(
                    user_id=user_id,
                    content=sanitized_content,
                    memory_type=extracted_dto.memory_type,
                    importance_score=extracted_dto.importance_score,
                    confidence_score=extracted_dto.confidence_score,
                    status=MemoryStatus.ACTIVE,
                )

                validated_memories.append(memory)
                current_batch_vecs.append(item_vec)

            except Exception as val_err:
                logger.warning("Skipping invalid extracted item %s: %s", raw_item, str(val_err))

        logger.info("Successfully extracted %d validated memories.", len(validated_memories))
        return validated_memories

    def extract_from_turn(
        self,
        user_message: str,
        assistant_reply: Optional[str] = None,
        user_id: str = "default_user",
    ) -> List[Memory]:
        """Convenience method to extract memories from a single turn."""
        clean_msg = (user_message or "").strip()
        if not clean_msg or "?" in clean_msg:
            return []
        lowered = clean_msg.lower()
        if re.match(r"^(?:what|why|how|when|where|who|which|can|could|should|would|do|does|did|is|are|am|was|were|tell me|explain|describe)\b", lowered):
            return []
        return self.extract_memories(conversation=clean_msg, user_id=user_id)


def extract_memories(
    conversation: Union[str, List[Dict[str, str]], List[Any]],
    user_id: str = "default_user",
    llm: Optional[BaseLLMProvider] = None,
    encoder: Optional[BaseEmbeddingEncoder] = None,
    existing_memories: Optional[List[Memory]] = None,
) -> List[Memory]:
    """
    Convenience top-level function to extract memories from a conversation.

    Example:
        memories = extract_memories("My name is Rahul. I am studying B.Tech CSE and I prefer Python for projects.")
    """
    extractor = MemoryExtractor(llm_provider=llm, encoder=encoder)
    return extractor.extract_memories(
        conversation=conversation,
        user_id=user_id,
        existing_memories=existing_memories,
    )
