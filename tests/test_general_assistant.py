"""
Comprehensive tests for General-Purpose AI Assistant + Resource-Efficient Long-Term Memory Layer:
1. English general questions (Ramayan, AI, ML, BFS/DFS, Python sorting, SQL)
2. Hindi questions (मशीन लर्निंग क्या है?, रामायण)
3. Hinglish questions (machine learning simple language me samjhao, RAG kya hota hai)
4. General question with zero memory injected
5. Relevant personal memory retrieved and augmented
6. Irrelevant personal memory filtered out by Relevance Gate
7. Unknown personal information not hallucinated
8. Natural follow-up questions with conversational context (What is RAG? -> Why is it useful?)
9. Dynamic LLM response instead of generic fallback
10. Token/memory resource efficiency metric behavior
"""
import pytest
from core.schemas import ChatRequest, MemoryCreate, MemoryType
from memory.manager import MemoryManager


def test_english_general_questions(memory_manager: MemoryManager):
    """General knowledge inquiries in English are answered naturally without memory injection."""
    queries = [
        ("Tell me about the Ramayan.", ["Ramayana", "Rama", "Valmiki"]),
        ("What is Artificial Intelligence?", ["Artificial Intelligence", "AI", "intelligence"]),
        ("Explain Machine Learning.", ["Machine Learning", "patterns", "data"]),
        ("Explain BFS and DFS.", ["Breadth-First", "Depth-First", "queue"]),
        ("Write a Python program for sorting.", ["def", "sort", "return"]),
        ("Explain SQL.", ["SQL", "relational", "database"]),
    ]

    for prompt, expected_keywords in queries:
        resp = memory_manager.process_chat(ChatRequest(
            message=prompt,
            user_id="user_en_gk",
            auto_extract_memories=False,
        ))
        assert resp.memory_used is False
        assert len(resp.injected_memories) == 0
        assert resp.context_tokens_used == 0
        assert "I can provide information on this topic using general knowledge" not in resp.reply
        assert any(k.lower() in resp.reply.lower() for k in expected_keywords)


def test_hindi_question_understanding(memory_manager: MemoryManager):
    """Hindi language inquiries are understood and answered in natural Hindi without memory injection."""
    queries = [
        ("मशीन लर्निंग क्या है?", ["मशीन लर्निंग", "आर्टिफिशियल इंटेलिजेंस", "डेटा"]),
        ("रामायण के बारे में बताएं", ["रामायण", "राम", "वाल्मीकि"]),
    ]

    for prompt, expected_keywords in queries:
        resp = memory_manager.process_chat(ChatRequest(
            message=prompt,
            user_id="user_hi_gk",
            auto_extract_memories=False,
        ))
        assert resp.memory_used is False
        assert len(resp.injected_memories) == 0
        assert resp.context_tokens_used == 0
        assert any(k in resp.reply for k in expected_keywords)


def test_hinglish_question_understanding(memory_manager: MemoryManager):
    """Hinglish inquiries are understood and answered in natural Hinglish style."""
    queries = [
        ("machine learning simple language me samjhao", ["Machine Learning", "data", "seekhte"]),
        ("RAG kya hota hai and iska use kyu karte hain?", ["RAG", "Retrieval-Augmented", "vector"]),
    ]

    for prompt, expected_keywords in queries:
        resp = memory_manager.process_chat(ChatRequest(
            message=prompt,
            user_id="user_hinglish_gk",
            auto_extract_memories=False,
        ))
        assert resp.memory_used is False
        assert len(resp.injected_memories) == 0
        assert resp.context_tokens_used == 0
        assert any(k.lower() in resp.reply.lower() for k in expected_keywords)


def test_general_question_zero_memory_injection(memory_manager: MemoryManager):
    """General knowledge query with existing stored memory must inject 0 memories."""
    # Store unrelated personal memory
    memory_manager.create_memory(MemoryCreate(
        content="User is a senior software engineer living in Bangalore.",
        user_id="distractor_user_1",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.9,
    ))

    # Query general science concept
    resp = memory_manager.process_chat(ChatRequest(
        message="What is photosynthesis?",
        user_id="distractor_user_1",
        auto_extract_memories=False,
    ))

    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert resp.context_tokens_used == 0
    assert resp.context_tokens_saved == 0
    assert "Bangalore" not in resp.reply


def test_relevant_personal_memory_injected(memory_manager: MemoryManager):
    """Personal question targeting stored fact must retrieve and inject memory into prompt."""
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Abhinandan.",
        user_id="personal_user_1",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.95,
    ))

    resp = memory_manager.process_chat(ChatRequest(
        message="What is my name?",
        user_id="personal_user_1",
        auto_extract_memories=False,
    ))

    assert resp.memory_used is True
    assert len(resp.injected_memories) >= 1
    assert "Abhinandan" in resp.reply
    assert "Your name is Abhinandan." in resp.reply


def test_irrelevant_personal_memory_filtered(memory_manager: MemoryManager):
    """Irrelevant personal memory must be filtered out for technical query."""
    memory_manager.create_memory(MemoryCreate(
        content="User's favorite sport is cricket.",
        user_id="cricket_fan_user",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.8,
    ))

    resp = memory_manager.process_chat(ChatRequest(
        message="Explain SQL.",
        user_id="cricket_fan_user",
        auto_extract_memories=False,
    ))

    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert "cricket" not in resp.reply.lower()
    assert "SQL" in resp.reply or "Structured Query" in resp.reply


def test_unknown_personal_information_not_hallucinated(memory_manager: MemoryManager):
    """Personal questions with no stored memory must state not stored and not hallucinate."""
    resp = memory_manager.process_chat(ChatRequest(
        message="What is my favorite programming language?",
        user_id="blank_slate_user",
        auto_extract_memories=False,
    ))

    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert "I don't have any specific active memories" in resp.reply or "not currently stored" in resp.reply
    assert "Your favorite programming language is Python" not in resp.reply


def test_natural_conversation_follow_up_context(memory_manager: MemoryManager):
    """Follow-up questions understand topic context using compact conversational history."""
    user_id = "followup_user"

    # Turn 1: Introduce topic
    t1 = memory_manager.process_chat(ChatRequest(
        message="What is RAG?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert "Retrieval-Augmented" in t1.reply or "RAG" in t1.reply

    # Turn 2: Follow-up question referring to "it"
    t2 = memory_manager.process_chat(ChatRequest(
        message="Why is it useful?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert t2.memory_used is False
    # Verifies understanding that "it" refers to RAG
    assert any(term in t2.reply.lower() for term in ["rag", "hallucination", "retrieval", "vector", "external"])


def test_dynamic_llm_response_not_generic_fallback(memory_manager: MemoryManager):
    """Verifies that no query ever triggers the rejected generic fallback sentence."""
    test_prompts = [
        "Tell me about the Ramayan.",
        "What is quantum computing?",
        "Explain Docker containers.",
        "How do transformers work in deep learning?",
    ]

    for p in test_prompts:
        resp = memory_manager.process_chat(ChatRequest(
            message=p,
            user_id="dynamic_check_user",
            auto_extract_memories=False,
        ))
        assert "I can provide information on this topic using general knowledge" not in resp.reply
        assert len(resp.reply) > 40


def test_token_and_memory_injection_behavior(memory_manager: MemoryManager):
    """Resource metrics reflect 0 memory tokens for GK and accurate savings for injected memory."""
    user_id = "token_validation_user"

    # 1. Create 3 active memories
    for i in range(3):
        memory_manager.create_memory(MemoryCreate(
            content=f"User background preference item #{i} for automated cloud deployments.",
            user_id=user_id,
            importance_score=0.85,
        ))

    # General Knowledge Query -> 0 memory tokens
    gk_resp = memory_manager.process_chat(ChatRequest(
        message="What is the speed of light?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert gk_resp.memory_used is False
    assert gk_resp.context_tokens_used == 0
    assert gk_resp.context_tokens_saved == 0
    assert gk_resp.context_reduction_percent == 0.0

    # Memory Query -> compact injection with positive token reduction
    mem_resp = memory_manager.process_chat(ChatRequest(
        message="What are my preferences for automated cloud deployments?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert mem_resp.memory_used is True
    assert len(mem_resp.injected_memories) >= 1
    assert mem_resp.context_tokens_used > 0
    assert mem_resp.context_tokens_saved >= 0


def test_ramayan_followup_yug_context(memory_manager: MemoryManager):
    """
    Regression Test:
    Turn 1: 'Ramayan ke bare me batao'
    Turn 2: 'ye kis yug me hua tha'
    Verifies that assistant understands 'ye' refers to the Ramayan and answers with Treta Yuga,
    without returning the generic Concept/Use Cases/Key Benefit template.
    """
    user_id = "ramayan_followup_tester"

    # Turn 1
    t1 = memory_manager.process_chat(ChatRequest(
        message="Ramayan ke bare me batao",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert t1.memory_used is False
    assert any(w in t1.reply.lower() for w in ["ramayan", "valmiki", "rama", "lord rama"])
    assert "1. **Concept**:" not in t1.reply

    # Turn 2: Follow-up using 'ye' and asking about 'yug'
    t2 = memory_manager.process_chat(ChatRequest(
        message="ye kis yug me hua tha",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert t2.memory_used is False
    assert len(t2.injected_memories) == 0
    assert t2.context_tokens_used == 0
    # Expected historical/traditional context: Treta Yug
    assert "treta" in t2.reply.lower() or "त्रेता" in t2.reply
    # Must NOT return generic placeholder template
    assert "1. **Concept**:" not in t2.reply
    assert "Yeh apne domain ka ek important topic hai" not in t2.reply
    assert "Use Cases" not in t2.reply


def test_current_affairs_prime_minister_no_generic_template(memory_manager: MemoryManager):
    """
    Regression Test:
    Query: 'prime minister kaun hai'
    Must answer question specifically (Narendra Modi for India), make clear real-time verification limitation,
    and NOT return the generic Concept / Use Cases / Key Benefit template.
    """
    resp = memory_manager.process_chat(ChatRequest(
        message="prime minister kaun hai",
        user_id="pm_query_user",
        auto_extract_memories=False,
    ))
    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert resp.context_tokens_used == 0
    # Answers the question
    assert "Narendra Modi" in resp.reply or "नरेंद्र मोदी" in resp.reply
    # States verification limitation
    assert any(term in resp.reply.lower() for term in ["live", "real-time", "search", "official", "internet"])
    # Generic templates MUST NOT be returned
    assert "1. **Concept**:" not in resp.reply
    assert "Yeh apne domain ka ek important topic hai" not in resp.reply
    assert "Use Cases" not in resp.reply
    assert "Key Benefit" not in resp.reply


def test_followup_question_variants_use_compact_context(memory_manager: MemoryManager):
    """
    Regression Test:
    Verifies that various follow-up forms ('Iska example do', 'Ye kaise kaam karta hai', 'Explain it more simply')
    correctly resolve the antecedent topic from recent context without returning generic templates.
    """
    user_id = "followup_variants_user"

    # Scenario 1: Machine Learning -> Iska example do
    memory_manager.process_chat(ChatRequest(
        message="Explain Machine Learning.",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    ex_resp = memory_manager.process_chat(ChatRequest(
        message="Iska example do.",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert ex_resp.memory_used is False
    assert any(term in ex_resp.reply.lower() for term in ["spam", "recommendation", "detection", "vision", "example"])
    assert "1. **Concept**:" not in ex_resp.reply

    # Scenario 2: RAG -> Ye kaise kaam karta hai?
    user_rag = "rag_workflow_user"
    memory_manager.process_chat(ChatRequest(
        message="RAG kya hota hai?",
        user_id=user_rag,
        auto_extract_memories=False,
    ))
    how_resp = memory_manager.process_chat(ChatRequest(
        message="Ye kaise kaam karta hai?",
        user_id=user_rag,
        auto_extract_memories=False,
    ))
    assert how_resp.memory_used is False
    assert any(term in how_resp.reply.lower() for term in ["vector", "retrieval", "chunk", "embedding", "database", "prompt"])
    assert "1. **Concept**:" not in how_resp.reply

    # Scenario 3: BFS and DFS -> Explain it more simply
    user_graph = "graph_user"
    memory_manager.process_chat(ChatRequest(
        message="Explain BFS and DFS.",
        user_id=user_graph,
        auto_extract_memories=False,
    ))
    simple_resp = memory_manager.process_chat(ChatRequest(
        message="Explain it more simply.",
        user_id=user_graph,
        auto_extract_memories=False,
    ))
    assert simple_resp.memory_used is False
    assert any(term in simple_resp.reply.lower() for term in ["ripple", "pond", "corridor", "maze", "level", "bfs", "queue", "stack"])
    assert "1. **Concept**:" not in simple_resp.reply


def test_no_generic_placeholder_templates_returned(memory_manager: MemoryManager):
    """
    Regression Test:
    Ensures that generic placeholder templates ('Concept', 'Use Cases', 'Key Benefit')
    are never returned for any general knowledge or technical query.
    """
    test_queries = [
        "Tell me about the Ramayan.",
        "prime minister kaun hai",
        "What is quantum computing?",
        "Explain Docker containers.",
        "What is photosynthesis?",
        "What is the speed of light?",
    ]

    for query in test_queries:
        resp = memory_manager.process_chat(ChatRequest(
            message=query,
            user_id="template_guard_user",
            auto_extract_memories=False,
        ))
        assert "1. **Concept**:" not in resp.reply
        assert "1. **Core Concept**:" not in resp.reply
        assert "Yeh apne domain ka ek important topic hai" not in resp.reply
        assert "2. **Use Cases**:" not in resp.reply
        assert "3. **Key Benefit**:" not in resp.reply
        assert len(resp.reply) > 50
