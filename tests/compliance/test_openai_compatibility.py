"""OpenAI SDK compliance tests."""

import pytest

# Note: These tests use the actual OpenAI SDK to verify compatibility
# They require the vLLM service to be running

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
@pytest.mark.compliance
def test_embedding_with_openai_sdk():
    """Test embedding endpoint using official OpenAI Python SDK."""
    # Configure to use local vLLM service
    client = openai.OpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )

    response = client.embeddings.create(
        input="Hello world",
        model="Qwen/Qwen3-Embedding-0.6B"
    )

    assert response.object == "list"
    assert len(response.data) == 1
    assert len(response.data[0].embedding) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
@pytest.mark.compliance
def test_completion_with_openai_sdk():
    """Test completion endpoint using official OpenAI Python SDK."""
    client = openai.OpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )

    response = client.completions.create(
        prompt="Once upon a time",
        model="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=50
    )

    assert response.object == "text_completion"
    assert len(response.choices) >= 1
    assert len(response.choices[0].text) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
@pytest.mark.compliance
def test_batch_embedding_with_openai_sdk():
    """Test batch embedding using OpenAI SDK."""
    client = openai.OpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )

    texts = ["text1", "text2", "text3"]
    response = client.embeddings.create(
        input=texts,
        model="Qwen/Qwen3-Embedding-0.6B"
    )

    assert len(response.data) == 3
    assert all(hasattr(item, "embedding") for item in response.data)


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
@pytest.mark.compliance
def test_completion_parameters_with_openai_sdk():
    """Test completion with various parameters using OpenAI SDK."""
    client = openai.OpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )

    response = client.completions.create(
        prompt="The quick brown",
        model="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=20,
        temperature=0.8,
        top_p=0.9
    )

    assert response.object == "text_completion"
    assert len(response.choices[0].text) > 0
