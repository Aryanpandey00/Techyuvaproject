"""
Tests for LLM provider architecture, selection, secure credentials,
mocked API responses, and strict error handling (no silent fallbacks).
"""
import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from core.exceptions import LLMProviderError
from core.schemas import ChatRequest, MemoryCreate, MemoryType
from llm.provider import (
    LocalHuggingFaceLLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
)
from memory.manager import MemoryManager


# ==============================================================================
# 1. Provider Selection and Configuration Tests
# ==============================================================================


def test_provider_factory_selection():
    """Verify factory returns appropriate provider instance without error."""
    mock_p = get_llm_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)

    openai_p = get_llm_provider("openai", api_key="sk-test-key-123")
    assert isinstance(openai_p, OpenAILLMProvider)
    assert openai_p.api_key == "sk-test-key-123"
    assert openai_p.model_name == "gpt-4o-mini"
    assert openai_p.base_url == "https://api.openai.com/v1"

    local_p = get_llm_provider("local")
    assert isinstance(local_p, LocalHuggingFaceLLMProvider)


def test_custom_model_and_base_url_configuration():
    """Verify custom model names and OpenAI-compatible base URLs (e.g. Groq, Ollama) are configured."""
    custom_p = get_llm_provider(
        "openai",
        api_key="gsk-test-key",
        model_name="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1/",
    )
    assert isinstance(custom_p, OpenAILLMProvider)
    assert custom_p.api_key == "gsk-test-key"
    assert custom_p.model_name == "llama-3.3-70b-versatile"
    # Trailing slash should be stripped cleanly
    assert custom_p.base_url == "https://api.groq.com/openai/v1"


def test_openai_reads_api_key_from_environment(monkeypatch):
    """Verify OpenAI provider automatically and securely reads credentials from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-secret-key-xyz")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.endpoint.com/v1")

    provider = OpenAILLMProvider()
    assert provider.api_key == "sk-env-secret-key-xyz"
    assert provider.model_name == "gpt-4o"
    assert provider.base_url == "https://custom.endpoint.com/v1"


# ==============================================================================
# 2. Missing Credentials Handling Tests
# ==============================================================================


def test_openai_missing_credentials_raises_clear_error(monkeypatch):
    """Verify OpenAILLMProvider raises a clear LLMProviderError when credentials are missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAILLMProvider(api_key=None)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_response("What is the speed of light?")
    assert "OpenAI API key is missing" in str(exc_info.value.message)

    with pytest.raises(LLMProviderError) as exc_info_extract:
        provider.extract_memories("My favorite dish is biryani.")
    assert "OpenAI API key is missing" in str(exc_info_extract.value.message)

    with pytest.raises(LLMProviderError) as exc_info_compress:
        provider.compress_memories(["fact 1", "fact 2"])
    assert "OpenAI API key is missing" in str(exc_info_compress.value.message)


# ==============================================================================
# 3. Mocked Successful API Responses (No Real API Keys / No Paid Calls)
# ==============================================================================


def test_openai_successful_chat_generation():
    """Verify response generation with mocked OpenAI-compatible REST endpoint."""
    provider = OpenAILLMProvider(api_key="sk-test-valid", model_name="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        reply = provider.generate_response(
            prompt="What is the speed of light?",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )

        assert "299,792,458 meters per second" in reply
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-valid"
        assert call_kwargs["json"]["model"] == "gpt-4o-mini"
        assert any(m["role"] == "user" and m["content"] == "What is the speed of light?" for m in call_kwargs["json"]["messages"])


def test_openai_successful_memory_extraction():
    """Verify memory extraction parses JSON from mocked OpenAI response."""
    provider = OpenAILLMProvider(api_key="sk-test-valid")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps([
                        {
                            "content": "User lives and works in Bengaluru.",
                            "memory_type": "personal",
                            "importance_score": 0.90,
                            "confidence_score": 1.0,
                        }
                    ]),
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        extracted = provider.extract_memories("I live and work in Bengaluru.")
        assert len(extracted) == 1
        assert extracted[0]["content"] == "User lives and works in Bengaluru."
        assert extracted[0]["memory_type"] == "personal"


def test_openai_successful_memory_compression():
    """Verify memory compression returns concise summary from mocked OpenAI response."""
    provider = OpenAILLMProvider(api_key="sk-test-valid")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "User is a senior backend engineer based in Bengaluru with expertise in Python.",
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        compressed = provider.compress_memories([
            "User is a software engineer.",
            "User has 7 years of backend experience.",
            "User lives in Bengaluru and uses Python.",
        ])
        assert "senior backend engineer based in Bengaluru" in compressed


def test_openai_successful_importance_scoring():
    """Verify importance score parsing from mocked OpenAI response."""
    provider = OpenAILLMProvider(api_key="sk-test-valid")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "0.85"}}]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        score = provider.score_importance("User has severe penicillin allergy.")
        assert score == 0.85


# ==============================================================================
# 4. API Error Handling (No Silent Fallback to Mock)
# ==============================================================================


def test_openai_401_unauthorized_error_handling():
    """Verify 401 Unauthorized raises LLMProviderError and does NOT fall back to mock."""
    provider = OpenAILLMProvider(api_key="sk-invalid-key")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"error": {"message": "Incorrect API key provided: sk-invalid-key."}}'
    mock_response.json.return_value = {
        "error": {"message": "Incorrect API key provided: sk-invalid-key."}
    }

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate_response("Test prompt")
        assert "401" in str(exc_info.value.message)
        assert "Incorrect API key" in str(exc_info.value.message)


def test_openai_429_rate_limit_error_handling():
    """Verify 429 Rate Limit raises LLMProviderError with quota explanation."""
    provider = OpenAILLMProvider(api_key="sk-test-key")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = '{"error": {"message": "You exceeded your current quota."}}'
    mock_response.json.return_value = {
        "error": {"message": "You exceeded your current quota."}
    }

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate_response("Test prompt")
        assert "429" in str(exc_info.value.message)
        assert "quota" in str(exc_info.value.message)


def test_openai_500_server_error_handling():
    """Verify 500 Internal Server Error raises LLMProviderError."""
    provider = OpenAILLMProvider(api_key="sk-test-key")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = '{"error": {"message": "The server had an internal error."}}'
    mock_response.json.return_value = {
        "error": {"message": "The server had an internal error."}
    }

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate_response("Test prompt")
        assert "500" in str(exc_info.value.message)


def test_openai_network_timeout_handling():
    """Verify network timeouts raise clear LLMProviderError."""
    provider = OpenAILLMProvider(api_key="sk-test-key", timeout=10.0)

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate_response("Test prompt")
        assert "timed out" in str(exc_info.value.message)


# ==============================================================================
# 5. End-to-End MemoryManager Integration & Preservation
# ==============================================================================


def test_memory_manager_with_openai_error_propagates_without_silent_mock_fallback():
    """Verify MemoryManager does not silently fall back to Mock when OpenAI fails."""
    manager = MemoryManager(
        llm_provider_type="openai",
        openai_api_key="sk-test-key",
        in_memory_db=True,
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"error": {"message": "Invalid API key."}}'
    mock_response.json.return_value = {"error": {"message": "Invalid API key."}}

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(LLMProviderError) as exc:
            manager.process_chat(ChatRequest(
                message="What is quantum computing?",
                user_id="user_test_no_fallback",
            ))
        assert "401" in str(exc.value.message)


def test_memory_manager_with_openai_preserves_retrieval_and_metrics():
    """
    Verify MemoryManager with OpenAI provider preserves:
    - Long-term memory retrieval
    - Relevance filtering
    - Resource metrics (tokens avoided, reduction percent, latency)
    - Follow-up conversational turns
    """
    manager = MemoryManager(
        llm_provider_type="openai",
        openai_api_key="sk-test-key",
        in_memory_db=True,
    )

    # Pre-populate memory
    manager.create_memory(MemoryCreate(
        user_id="user_ai",
        content="User's preferred programming language is Rust.",
        memory_type=MemoryType.PREFERENCE,
        importance_score=0.95,
        confidence_score=1.0,
    ))

    # Mock response from OpenAI
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Your preferred programming language is Rust.",
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        resp = manager.process_chat(ChatRequest(
            message="What programming language do I prefer?",
            user_id="user_ai",
            relevance_threshold=0.60,
        ))

        # Check response content
        assert "Rust" in resp.reply
        assert resp.memory_used is True
        assert len(resp.injected_memories) == 1
        assert "Rust" in resp.injected_memories[0].memory.content

        # Verify resource metrics are preserved
        assert resp.context_tokens_used > 0
        assert resp.context_tokens_saved >= 0
        assert resp.context_reduction_percent >= 0.0
        assert resp.retrieval_latency_ms > 0

        # Verify payload sent to OpenAI contains retrieved context in system message
        call_kwargs = mock_post.call_args[1]
        sent_messages = call_kwargs["json"]["messages"]
        system_msg = next(m["content"] for m in sent_messages if m["role"] == "system")
        assert "Relevant Stored Memories" in system_msg
        assert "Rust" in system_msg


def test_local_provider_raises_clear_error_when_pipeline_fails():
    """Verify LocalHuggingFaceLLMProvider raises LLMProviderError when pipeline fails rather than returning mock."""
    local_p = LocalHuggingFaceLLMProvider(model_name="nonexistent-model")

    with pytest.raises(LLMProviderError) as exc_info:
        local_p.generate_response("Hello local model")
    assert "Failed to load local HuggingFace model" in str(exc_info.value.message)
