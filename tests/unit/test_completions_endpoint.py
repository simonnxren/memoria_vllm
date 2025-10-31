"""Unit tests for completions endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_generation_client():
    """Mock VLLMGenerationClient."""
    with patch("src.completions.generation_client") as mock:
        yield mock


def test_simple_completion(mock_generation_client, mock_vllm_completion_response):
    """Test simple text completion."""
    mock_generation_client.complete = AsyncMock(return_value=mock_vllm_completion_response)

    request_data = {
        "prompt": "Once upon a time",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "max_tokens": 50
    }

    response = client.post("/v1/completions", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert "choices" in data
    assert len(data["choices"]) >= 1
    assert "text" in data["choices"][0]
    assert "usage" in data


def test_completion_with_temperature(mock_generation_client, mock_vllm_completion_response):
    """Test completion with temperature parameter."""
    mock_generation_client.complete = AsyncMock(return_value=mock_vllm_completion_response)

    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "temperature": 0.5,
        "max_tokens": 10
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 200


def test_completion_with_top_p(mock_generation_client, mock_vllm_completion_response):
    """Test completion with top_p parameter."""
    mock_generation_client.complete = AsyncMock(return_value=mock_vllm_completion_response)

    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "top_p": 0.9,
        "max_tokens": 10
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 200


def test_completion_with_stop_sequences(mock_generation_client, mock_vllm_completion_response):
    """Test completion with stop sequences."""
    mock_generation_client.complete = AsyncMock(return_value=mock_vllm_completion_response)

    request_data = {
        "prompt": "Count: 1, 2, 3",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "max_tokens": 20,
        "stop": ["\n", "."]
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 200


def test_invalid_temperature():
    """Test that invalid temperature is rejected."""
    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "temperature": 5.0  # Invalid: > 2.0
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 422


def test_invalid_top_p():
    """Test that invalid top_p is rejected."""
    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "top_p": 1.5  # Invalid: > 1.0
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 422


def test_invalid_max_tokens():
    """Test that invalid max_tokens is rejected."""
    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "max_tokens": -1  # Invalid: < 1
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 422


def test_empty_prompt():
    """Test that empty prompt is rejected."""
    request_data = {
        "prompt": "",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8"
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 422


def test_service_unavailable(mock_generation_client):
    """Test handling of vLLM service unavailability."""
    from src.utils import VLLMConnectionError

    mock_generation_client.complete = AsyncMock(
        side_effect=VLLMConnectionError("Service unavailable")
    )

    request_data = {
        "prompt": "test prompt",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8"
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 503
    assert "detail" in response.json()
    assert "error" in response.json()["detail"]


def test_multiple_completions(mock_generation_client):
    """Test generating multiple completions (n parameter)."""
    mock_response = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1699123456,
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "choices": [
            {"text": " completion 1", "index": 0, "finish_reason": "length"},
            {"text": " completion 2", "index": 1, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    }
    mock_generation_client.complete = AsyncMock(return_value=mock_response)

    request_data = {
        "prompt": "Hello",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "n": 2,
        "max_tokens": 10
    }

    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"]) == 2
