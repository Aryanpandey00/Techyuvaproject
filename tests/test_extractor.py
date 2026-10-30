"""
Unit tests for the Memory Extraction System:
- Extraction of atomic facts, preferences, and profile details
- Schema validation (content, memory_type, importance_score, confidence_score)
- Rejection of greetings and casual conversation
- Rejection of temporary / ephemeral remarks
- Semantic duplicate detection
- Sensitive information redaction
- Mocked LLM response verification
- Independent callable interface
"""
import json
from unittest.mock import MagicMock
import pytest
from core.schemas import Memory, MemoryStatus, MemoryType
from embeddings.encoder import MockEmbeddingEncoder
from llm.base import BaseLLMProvider
from memory.extractor import MemoryExtractor, extract_memories


def test_extract_memories_rahul_example(mock_encoder: MockEmbeddingEncoder):
    """
    Test the exact prompt example:
    'My name is Rahul. I am studying B.Tech CSE and I prefer Python for projects.'
    """
    conversation = "My name is Rahul. I am studying B.Tech CSE and I prefer Python for projects."
    memories = extract_memories(conversation=conversation, encoder=mock_encoder)

    assert len(memories) >= 3

    contents = [m.content for m in memories]
    assert any("Rahul" in c for c in contents)
    assert any("B.Tech CSE" in c or "studying" in c for c in contents)
    assert any("Python" in c or "prefer" in c for c in contents)

    for m in memories:
        assert m.content is not None
        assert m.memory_type in MemoryType
        assert 0.0 <= m.importance_score <= 1.0
        assert 0.0 <= m.confidence_score <= 1.0
        assert m.status == MemoryStatus.ACTIVE


def test_reject_greetings_and_casual_banter(mock_encoder: MockEmbeddingEncoder):
    """Test that casual greetings and social chatter are rejected."""
    casual_inputs = [
        "Hello! How are you doing today?",
        "Good morning!",
        "Hey what's up?",
        "Thank you so much, bye!",
        "Cool, got it, thanks.",
    ]

    for utterance in casual_inputs:
        memories = extract_memories(conversation=utterance, encoder=mock_encoder)
        assert len(memories) == 0, f"Expected 0 memories for banter: '{utterance}', got {len(memories)}"


def test_reject_temporary_irrelevant_information(mock_encoder: MockEmbeddingEncoder):
    """Test that temporary/ephemeral remarks are rejected."""
    temp_inputs = [
        "I'm eating a sandwich right now.",
        "brb in 5 mins",
        "heading out for lunch",
    ]

    for utterance in temp_inputs:
        memories = extract_memories(conversation=utterance, encoder=mock_encoder)
        assert len(memories) == 0, f"Expected 0 memories for temporary utterance: '{utterance}'"


def test_duplicate_memory_detection(mock_encoder: MockEmbeddingEncoder):
    """Test that semantically redundant memories are rejected."""
    extractor = MemoryExtractor(encoder=mock_encoder, duplicate_similarity_threshold=0.85)

    existing = [
        Memory(
            content="User prefers Python for projects.",
            memory_type=MemoryType.PREFERENCE,
            importance_score=0.9,
            confidence_score=1.0,
            status=MemoryStatus.ACTIVE,
        )
    ]

    # Try extracting identical statement
    extracted = extractor.extract_memories(
        conversation="I prefer Python for projects.",
        existing_memories=existing,
    )

    # Should be detected as duplicate and rejected
    assert len(extracted) == 0


def test_sensitive_information_sanitization(mock_encoder: MockEmbeddingEncoder):
    """Test that passwords and secret tokens are sanitized/redacted."""
    extractor = MemoryExtractor(encoder=mock_encoder)

    # Mock LLM returning text with secret
    mock_llm = MagicMock(spec=BaseLLMProvider)
    mock_llm.generate_response.return_value = json.dumps([
        {
            "content": "User's password is: secret12345! and apiKey is sk-abcdef1234567890abcdef1234567890",
            "memory_type": "personal",
            "importance_score": 0.8,
            "confidence_score": 0.9,
        }
    ])

    extractor.llm = mock_llm
    memories = extractor.extract_memories("Here is my login info")

    assert len(memories) == 1
    assert "secret12345!" not in memories[0].content
    assert "[REDACTED_PASSWORD]" in memories[0].content or "[REDACTED_SECRET_KEY]" in memories[0].content


def test_structured_json_mocked_llm_response(mock_encoder: MockEmbeddingEncoder):
    """Test extractor with custom mocked LLM returning markdown json block."""
    mock_llm = MagicMock(spec=BaseLLMProvider)
    mock_llm.generate_response.return_value = """
    Here are the extracted memories:
    ```json
    [
      {
        "content": "User lives in Bangalore, India.",
        "memory_type": "personal",
        "importance_score": 0.95,
        "confidence_score": 1.0
      },
      {
        "content": "User is building an autonomous agent project.",
        "memory_type": "project",
        "importance_score": 0.90,
        "confidence_score": 0.95
      }
    ]
    ```
    """

    extractor = MemoryExtractor(llm_provider=mock_llm, encoder=mock_encoder)
    memories = extractor.extract_memories("I am based in Bangalore and working on an autonomous agent project.")

    assert len(memories) == 2
    assert memories[0].content == "User lives in Bangalore, India."
    assert memories[0].memory_type == MemoryType.PERSONAL
    assert memories[1].content == "User is building an autonomous agent project."
    assert memories[1].memory_type == MemoryType.PROJECT


def test_conversation_list_input_format(mock_encoder: MockEmbeddingEncoder):
    """Test extractor accepts conversation as a list of turn dictionaries."""
    conv_history = [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "My name is Priya and my goal is to learn Rust."},
    ]

    memories = extract_memories(conversation=conv_history, encoder=mock_encoder)
    assert len(memories) >= 1
    assert any("Priya" in m.content or "Rust" in m.content for m in memories)
