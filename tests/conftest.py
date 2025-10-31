"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_vllm_embedding_response():
    """Mock successful embedding response from vLLM."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1024,
                "index": 0
            }
        ],
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def mock_vllm_completion_response():
    """Mock successful completion response from vLLM."""
    return {
        "id": "cmpl-test123",
        "object": "text_completion",
        "created": 1699123456,
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "choices": [
            {
                "text": " there was a brave knight who embarked on a quest.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 12,
            "total_tokens": 17
        }
    }


@pytest.fixture
def mock_health_response():
    """Mock successful health check response."""
    return {
        "status": "ok"
    }
