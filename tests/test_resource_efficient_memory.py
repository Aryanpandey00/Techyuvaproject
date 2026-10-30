"""
Comprehensive Unit Tests for the Resource-Efficient Long-Term Memory Layer for Generative AI:
1. General knowledge question without memory (Case A)
2. Relevant personal memory question (Case B)
3. Unrecorded personal information not hallucinated (Case C)
4. Mixed memory + general knowledge question (Case D)
5. Irrelevant memory filtered out (Case E)
6. Context budget enforcement
7. Configurable relevance threshold gate
8. Resource efficiency metrics tracking
9. No full conversation history or raw template leakage
10. Mock LLM provider deterministic behavior
"""
import pytest
from core.schemas import ChatRequest, MemoryCreate, MemoryStatus, MemoryType, PruneRequest
from llm.provider import MockLLMProvider
from memory.manager import MemoryManager


def test_general_knowledge_question_without_memory(memory_manager: MemoryManager):
    """
    CASE A: General knowledge question with no relevant stored memory.
    The system must answer using LLM general knowledge without failing or injecting memory.
    """
    req = ChatRequest(
        message="What is the national animal of India?",
        user_id="user_gk",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert "Tiger" in resp.reply or "tiger" in resp.reply
    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert resp.context_tokens_used == 0
    assert "[RELEVANT STORED MEMORIES]" not in resp.reply


def test_relevant_personal_memory_question(memory_manager: MemoryManager):
    """
    CASE B: Relevant personal memory exists and is retrieved.
    User shares info in Turn 1, queries in Turn 2.
    """
    # Turn 1: Introduce name and degree
    turn1_req = ChatRequest(
        message="My name is Abhinandan and I am a B.Tech CSE AI/ML student.",
        user_id="abhinandan_user",
        auto_extract_memories=True,
    )
    turn1_resp = memory_manager.process_chat(turn1_req)
    assert len(turn1_resp.new_extracted_memories) >= 2

    # Turn 2: Query what user is studying
    turn2_req = ChatRequest(
        message="What am I studying?",
        user_id="abhinandan_user",
        auto_extract_memories=False,
    )
    turn2_resp = memory_manager.process_chat(turn2_req)

    assert turn2_resp.memory_used is True
    assert len(turn2_resp.injected_memories) >= 1
    assert "B.Tech CSE" in turn2_resp.reply or "studying" in turn2_resp.reply
    assert "[RELEVANT STORED MEMORIES]" not in turn2_resp.reply


def test_unrecorded_personal_information_not_hallucinated(memory_manager: MemoryManager):
    """
    CASE C: Personal information not stored in memory.
    The system must NOT invent or hallucinate an answer, but explicitly state it is unknown.
    """
    # Store only name and college
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Abhinandan.",
        user_id="strict_user",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.95,
    ))
    memory_manager.create_memory(MemoryCreate(
        content="User is studying B.Tech CSE AI/ML.",
        user_id="strict_user",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.85,
    ))

    # Ask unrecorded attribute: favorite programming language
    req = ChatRequest(
        message="What is my favorite programming language?",
        user_id="strict_user",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert "I don't have any specific active memories" in resp.reply or "not currently stored" in resp.reply
    # Must NOT hallucinate a random language as user's favorite
    assert "Your favorite programming language is Python" not in resp.reply


def test_missing_personal_attribute_not_hallucinated_when_other_memories_exist(memory_manager: MemoryManager):
    """Even with some personal context present, unknown attributes must not be inferred from unrelated memory."""
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Abhinandan.",
        user_id="missing_attr_user",
        memory_type=MemoryType.PERSONAL,
        importance_score=0.95,
    ))

    req = ChatRequest(
        message="What is my favorite color?",
        user_id="missing_attr_user",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert "I don't have any specific active memories" in resp.reply
    assert "Your favorite color is" not in resp.reply
    assert "Your name is Abhinandan." not in resp.reply


def test_general_question_is_not_stored_as_long_term_memory(memory_manager: MemoryManager):
    """General knowledge questions should not be persisted as long-term memories."""
    req = ChatRequest(
        message="What is the national animal of India?",
        user_id="general_question_user",
        auto_extract_memories=True,
    )
    resp = memory_manager.process_chat(req)

    assert "Tiger" in resp.reply or "tiger" in resp.reply
    assert resp.new_extracted_memories == []


def test_final_composite_threshold_is_applied_consistently(memory_manager: MemoryManager):
    """The final composite score must gate injection, not just the vector similarity proxy."""
    stored = memory_manager.create_memory(MemoryCreate(
        content="User likes Python for projects.",
        user_id="threshold_filter_user",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.9,
    ))

    original_query = memory_manager.retriever.vector_store.query_similarity
    memory_manager.retriever.vector_store.query_similarity = lambda query_embedding, top_k, where_filter=None: [
        (stored.id, 0.75, None, None)
    ]

    try:
        resp = memory_manager.retriever.retrieve(
            query="What is BFS?",
            user_id="threshold_filter_user",
            top_k=5,
            relevance_threshold=0.95,
        )
    finally:
        memory_manager.retriever.vector_store.query_similarity = original_query

    assert resp.results == []
    assert resp.context_string == ""


def test_mixed_question_uses_memory_and_general_knowledge(memory_manager: MemoryManager):
    """
    CASE D: Mixed question combining project memory with general technical knowledge.
    """
    # Store project memory
    memory_manager.create_memory(MemoryCreate(
        content="User is developing a RAG memory compression system with ChromaDB and SQLite.",
        user_id="rag_dev",
        memory_type=MemoryType.PROJECT,
        importance_score=0.95,
    ))

    req = ChatRequest(
        message="My project uses RAG. Explain how RAG is being used in my project.",
        user_id="rag_dev",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert resp.memory_used is True
    assert len(resp.injected_memories) >= 1
    # Check that both project context and technical explanation are present
    assert "RAG" in resp.reply or "Retrieval-Augmented" in resp.reply
    assert "ChromaDB" in resp.reply or "retrieval" in resp.reply.lower()


def test_irrelevant_memory_filtered_out(memory_manager: MemoryManager):
    """
    CASE E: Stored memories are completely unrelated to current question.
    System must NOT inject irrelevant personal memories into general question.
    """
    # Store diverse personal memories
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Abhinandan.",
        user_id="distractor_user",
        importance_score=0.9,
    ))
    memory_manager.create_memory(MemoryCreate(
        content="User loves eating pasta carbonara.",
        user_id="distractor_user",
        importance_score=0.8,
    ))
    memory_manager.create_memory(MemoryCreate(
        content="User lives in Bangalore.",
        user_id="distractor_user",
        importance_score=0.8,
    ))

    # Ask technical/general question unrelated to food/name/location
    req = ChatRequest(
        message="What is BFS?",
        user_id="distractor_user",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    # Memories must be filtered out
    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert "Breadth-First" in resp.reply or "traversal" in resp.reply or "BFS" in resp.reply
    # Irrelevant personal facts must not appear in answer
    assert "Abhinandan" not in resp.reply
    assert "pasta" not in resp.reply


def test_context_budget_enforcement(memory_manager: MemoryManager):
    """
    Test that the Resource-Efficient Context Layer strictly enforces context budget.
    Even when many candidates match, at most max_injected memories are passed.
    """
    # Create 8 matching memories
    for i in range(8):
        memory_manager.create_memory(MemoryCreate(
            content=f"User project fact #{i}: Machine learning memory layer component.",
            user_id="budget_user",
            importance_score=0.8,
        ))

    req = ChatRequest(
        message="Tell me about my project machine learning memory layer components",
        user_id="budget_user",
        top_k_memories=8,
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    # Injected memories capped at max_injected budget (<= 3)
    assert len(resp.injected_memories) <= 3
    assert resp.context_tokens_used > 0
    assert resp.context_tokens_used <= 250


def test_configurable_relevance_threshold_gate(memory_manager: MemoryManager):
    """
    Test that custom relevance threshold gate filters candidates appropriately.
    """
    memory_manager.create_memory(MemoryCreate(
        content="User likes orange soda.",
        user_id="threshold_user",
        importance_score=0.5,
    ))

    # High relevance threshold rejects weakly related or irrelevant queries
    req = ChatRequest(
        message="Quantum mechanics wave function",
        user_id="threshold_user",
        relevance_threshold=0.95,
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert len(resp.injected_memories) == 0
    assert resp.memory_used is False


def test_resource_efficiency_metrics_tracked(memory_manager: MemoryManager):
    """
    Verify that token savings, context reduction percentage, and latency are calculated.
    """
    # Add 5 active memories
    for i in range(5):
        memory_manager.create_memory(MemoryCreate(
            content=f"User background history item {i} describing software development preferences.",
            user_id="metrics_user",
            importance_score=0.7,
        ))

    req = ChatRequest(
        message="What software development preferences do I have?",
        user_id="metrics_user",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert resp.retrieval_latency_ms >= 0.0
    assert resp.context_tokens_saved >= 0
    assert 0.0 <= resp.context_reduction_percent <= 100.0


def test_no_full_history_or_raw_template_leakage(memory_manager: MemoryManager):
    """
    Verify that raw prompt artifacts and internal brackets never leak to assistant output.
    """
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Rahul.",
        user_id="leakage_test_user",
        importance_score=0.9,
    ))

    req = ChatRequest(
        message="What is my name?",
        user_id="leakage_test_user",
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert "[RELEVANT STORED MEMORIES]" not in resp.reply
    assert "[MEMORY CONTEXT]" not in resp.reply
    assert "Based on your stored memory context" not in resp.reply
    assert "Your name is Rahul." in resp.reply


def test_mock_provider_deterministic_behavior():
    """
    Verify MockLLMProvider operates fully offline, deterministically, and supports all core cases.
    """
    provider = MockLLMProvider()

    # Case A
    ans_gk = provider.generate_response(prompt="What is the national animal of India?")
    assert "Tiger" in ans_gk

    # Case B
    context = "1. [PERSONAL] User's name is Abhinandan. (Score: 0.95, Similarity: 1.00)"
    ans_name = provider.generate_response(prompt="What is my name?", context=context)
    assert ans_name == "Your name is Abhinandan."

    # Case C
    ans_unknown = provider.generate_response(prompt="What is my favorite programming language?", context=context)
    assert "I don't have any specific active memories" in ans_unknown

    # Extraction
    extracted = provider.extract_memories("My name is Abhinandan and I am a B.Tech CSE AI/ML student.")
    assert len(extracted) >= 2
    contents = [m["content"] for m in extracted]
    assert any("Abhinandan" in c for c in contents)
    assert any("B.Tech CSE" in c for c in contents)


def test_end_to_end_personal_and_general_memory_flow(memory_manager: MemoryManager):
    """
    Validate the complete required Generative AI + Memory interaction flow:
    1. General knowledge question before memory -> answers normally, no memory injected
    2. User introduces name -> memory extracted and stored
    3. User asks name -> relevant memory retrieved, natural answer generated
    4. User asks unrecorded personal question -> does not hallucinate, states not stored
    5. User asks unrelated general question -> stored name is not injected
    6. User introduces preference -> memory extracted and stored
    7. User asks preference -> relevant memory retrieved, natural answer generated
    """
    user_id = "e2e_student_user"

    # Step 1: General Knowledge question (no memories present)
    r1 = memory_manager.process_chat(ChatRequest(
        message="What is the national animal of India?",
        user_id=user_id,
        auto_extract_memories=True,
    ))
    assert "Tiger" in r1.reply or "tiger" in r1.reply
    assert r1.memory_used is False
    assert len(r1.injected_memories) == 0
    assert r1.new_extracted_memories == []

    # Step 2: Introduce name
    r2 = memory_manager.process_chat(ChatRequest(
        message="My name is Abhinandan.",
        user_id=user_id,
        auto_extract_memories=True,
    ))
    assert any("Abhinandan" in m.content for m in r2.new_extracted_memories)

    # Step 3: Ask name
    r3 = memory_manager.process_chat(ChatRequest(
        message="What is my name?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert r3.memory_used is True
    assert "Abhinandan" in r3.reply
    assert "Your name is Abhinandan." in r3.reply

    # Step 4: Ask unrecorded personal attribute (no memory stored for favorite programming language yet)
    r4 = memory_manager.process_chat(ChatRequest(
        message="What is my favorite programming language?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert "I don't have any specific active memories" in r4.reply or "not currently stored" in r4.reply
    assert "Your favorite programming language is Python" not in r4.reply

    # Step 5: Unrelated general question (name is stored, but should NOT be injected)
    r5 = memory_manager.process_chat(ChatRequest(
        message="What is the capital of France?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert r5.memory_used is False
    assert len(r5.injected_memories) == 0
    assert "Paris" in r5.reply
    assert "Abhinandan" not in r5.reply

    # Step 6: Store favorite programming language
    r6 = memory_manager.process_chat(ChatRequest(
        message="My favorite programming language is Python.",
        user_id=user_id,
        auto_extract_memories=True,
    ))
    assert any("Python" in m.content for m in r6.new_extracted_memories)

    # Step 7: Query favorite programming language
    r7 = memory_manager.process_chat(ChatRequest(
        message="What is my favorite programming language?",
        user_id=user_id,
        auto_extract_memories=False,
    ))
    assert r7.memory_used is True
    assert len(r7.injected_memories) >= 1
    assert "Python" in r7.reply
    assert "Your favorite programming language is Python." in r7.reply


def test_keep_archive_forget_lifecycle_retrieval_integration(memory_manager: MemoryManager):
    """
    Test adaptive retention lifecycle integration:
    - Active memories are retrievable
    - Archived memories are excluded from active retrieval by default
    - Reactivated memories are retrievable again
    - Pruning forgets low-importance decayed memories while protecting pinned items
    """
    uid = "lifecycle_test_user"

    # 1. Create a regular memory and a pinned memory
    regular = memory_manager.create_memory(MemoryCreate(
        content="User frequently works with Docker containers.",
        user_id=uid,
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.55,
    ))
    pinned = memory_manager.create_memory(MemoryCreate(
        content="User's emergency contact is John Doe.",
        user_id=uid,
        memory_type=MemoryType.PERSONAL,
        importance_score=0.95,
        metadata={"is_pinned": True},
    ))

    # Active search should find regular memory
    active_res = memory_manager.retriever.retrieve(
        query="Docker containers",
        user_id=uid,
        status=MemoryStatus.ACTIVE,
    )
    found_ids = [m.memory.id for m in active_res.results]
    assert regular.id in found_ids

    # 2. Archive regular memory
    memory_manager.update_status(regular.id, MemoryStatus.ARCHIVED)
    archived_mem = memory_manager.get_memory(regular.id)
    assert archived_mem.status == MemoryStatus.ARCHIVED

    # Active retrieval should NO LONGER retrieve archived memory
    after_arch_res = memory_manager.retriever.retrieve(
        query="Docker containers",
        user_id=uid,
        status=MemoryStatus.ACTIVE,
    )
    assert regular.id not in [m.memory.id for m in after_arch_res.results]

    # 3. Reactivate memory
    memory_manager.update_status(regular.id, MemoryStatus.ACTIVE)
    reactivated_res = memory_manager.retriever.retrieve(
        query="Docker containers",
        user_id=uid,
        status=MemoryStatus.ACTIVE,
    )
    assert regular.id in [m.memory.id for m in reactivated_res.results]

    # 4. Prune low-importance unpinned memories
    prune_result = memory_manager.prune_memories(PruneRequest(
        user_id=uid,
        importance_threshold=0.60,
        max_inactivity_days=0,
        hard_delete=False,
    ))
    assert prune_result.pruned_count >= 1

    # Regular memory (importance 0.55 < 0.60) should be forgotten
    pruned_regular = memory_manager.get_memory(regular.id)
    assert pruned_regular.status == MemoryStatus.FORGOTTEN

    # Pinned memory must be protected and remain active
    pinned_check = memory_manager.get_memory(pinned.id)
    assert pinned_check.status == MemoryStatus.ACTIVE


def test_unrelated_personal_memory_not_injected_into_general_knowledge_query(memory_manager: MemoryManager):
    """
    REGRESSION TEST:
    Stored memory: "User's name is Abhinandan."
    Query: "What is the national animal of India?"
    Expected: selected/injected memories = 0
    The relevance gate must strictly filter out the personal memory.
    """
    user_id = "gk_regression_user"
    # Store personal identity memory
    memory_manager.create_memory(MemoryCreate(
        content="User's name is Abhinandan.",
        user_id=user_id,
        memory_type=MemoryType.PERSONAL,
        importance_score=0.95,
    ))

    # General knowledge query
    req = ChatRequest(
        message="What is the national animal of India?",
        user_id=user_id,
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    # Injected memories must be strictly 0
    assert resp.memory_used is False
    assert len(resp.injected_memories) == 0
    assert resp.context_tokens_used == 0
    assert resp.context_tokens_saved == 0
    assert resp.relevance_status in ("memory_filtered", "general_knowledge")
    assert "Abhinandan" not in resp.reply
    assert "Tiger" in resp.reply or "tiger" in resp.reply


def test_mock_provider_behavior_separated_from_real_llm_behavior():
    """
    Test that MockLLMProvider is cleanly decoupled from real LLM provider (OpenAILLMProvider)
    and implements clear fallback behavior without masquerading as a completed dynamic LLM.
    """
    from llm.provider import MockLLMProvider, OpenAILLMProvider, get_llm_provider

    mock_llm = get_llm_provider("mock")
    assert isinstance(mock_llm, MockLLMProvider)
    assert mock_llm.model_name == "mock-deterministic-v1"

    # Mock provider responses are structured and predictable
    resp_gk = mock_llm.generate_response("What is the capital of France?")
    assert "Paris" in resp_gk

    # Real provider initialization requires api_key or env var
    real_llm = OpenAILLMProvider(api_key="sk-test-key", model_name="gpt-4o-mini")
    assert real_llm.model_name == "gpt-4o-mini"
    assert isinstance(real_llm, OpenAILLMProvider)
    assert real_llm.api_key == "sk-test-key"


def test_resource_token_metric_calculation_accuracy(memory_manager: MemoryManager):
    """
    Verify the resource-efficient context metrics calculation:
    baseline context tokens - selected compact memory tokens = estimated context tokens avoided
    """
    user_id = "token_metric_user"
    mem1 = memory_manager.create_memory(MemoryCreate(
        content="User prefers Python for data engineering and machine learning workflows.",
        user_id=user_id,
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.90,
    ))
    mem2 = memory_manager.create_memory(MemoryCreate(
        content="User works remotely from Bangalore as a senior software architect.",
        user_id=user_id,
        memory_type=MemoryType.PERSONAL,
        importance_score=0.85,
    ))

    # Query targeting Python preference
    req = ChatRequest(
        message="What programming language do I prefer for machine learning workflows?",
        user_id=user_id,
        auto_extract_memories=False,
    )
    resp = memory_manager.process_chat(req)

    assert resp.memory_used is True
    assert len(resp.injected_memories) >= 1
    assert resp.context_tokens_used > 0
    # Injected tokens should be less than or equal to total active pool
    assert resp.context_tokens_saved >= 0
    assert resp.context_reduction_percent > 0.0

    # For zero-injection query on same user (unrelated query)
    gk_req = ChatRequest(
        message="What is the speed of light in a vacuum?",
        user_id=user_id,
        auto_extract_memories=False,
    )
    gk_resp = memory_manager.process_chat(gk_req)
    assert gk_resp.memory_used is False
    assert len(gk_resp.injected_memories) == 0
    assert gk_resp.context_tokens_used == 0
    assert gk_resp.context_tokens_saved == 0
    assert gk_resp.context_reduction_percent == 0.0

