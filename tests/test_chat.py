"""
Unit tests for the Conversational Chat Response Generation & Memory Integration:
- User introduction and memory extraction turn
- Query turn with memory retrieval context
- Natural assistant response generation without template placeholders
- Second-person formulation of stored memories
"""
from core.schemas import ChatRequest
from memory.manager import MemoryManager


def test_chat_memory_retrieval_and_natural_response(memory_manager: MemoryManager):
    """
    Test the complete conversational memory flow:
    1. User: 'my name is abhinandan' -> extracts & stores memory
    2. User: 'what is my name' -> retrieves stored memory and replies 'Your name is Abhinandan.'
    """
    # Turn 1: Introduce name
    turn1_req = ChatRequest(
        message="my name is abhinandan",
        user_id="user_abhinandan",
        auto_extract_memories=True,
    )
    turn1_resp = memory_manager.process_chat(turn1_req)
    assert len(turn1_resp.new_extracted_memories) >= 1
    assert "Abhinandan" in turn1_resp.new_extracted_memories[0].content

    # Turn 2: Query name
    turn2_req = ChatRequest(
        message="what is my name",
        user_id="user_abhinandan",
        auto_extract_memories=False,
    )
    turn2_resp = memory_manager.process_chat(turn2_req)

    # Verify natural response
    assert turn2_resp.reply == "Your name is Abhinandan."
    # Verify no template placeholders leaked
    assert "[RELEVANT STORED MEMORIES]" not in turn2_resp.reply
    assert "[MEMORY CONTEXT]" not in turn2_resp.reply
    assert "Based on your stored memory context" not in turn2_resp.reply


def test_chat_preferences_response(memory_manager: MemoryManager):
    """Test natural response for preference-related questions."""
    # Store food preference
    memory_manager.process_chat(ChatRequest(
        message="My favorite dish is pasta carbonara.",
        user_id="foodie_user",
        auto_extract_memories=True,
    ))

    # Query preference
    resp = memory_manager.process_chat(ChatRequest(
        message="What is my favorite dish?",
        user_id="foodie_user",
    ))

    assert "Your favorite dish is pasta carbonara." in resp.reply
    assert "[RELEVANT STORED MEMORIES]" not in resp.reply
    assert "[MEMORY CONTEXT]" not in resp.reply


def test_chat_unknown_query_response(memory_manager: MemoryManager):
    """Test appropriate response when no relevant memory exists yet."""
    resp = memory_manager.process_chat(ChatRequest(
        message="What is my name?",
        user_id="brand_new_user",
    ))
    assert "I don't have any specific active memories" in resp.reply
    assert "[RELEVANT STORED MEMORIES]" not in resp.reply
