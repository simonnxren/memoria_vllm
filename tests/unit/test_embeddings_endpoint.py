"""Unit tests for embeddings endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_embedding_client():
    """Mock VLLMEmbeddingClient."""
    with patch("src.embeddings.embedding_client") as mock:
        yield mock


def test_single_text_embedding(mock_embedding_client, mock_vllm_embedding_response):
    """Test embedding generation for single text."""
    mock_embedding_client.embed = AsyncMock(return_value=mock_vllm_embedding_response)

    request_data = {
        "input": "hello world",
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }

    response = client.post("/v1/embeddings", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 1024
    assert data["data"][0]["index"] == 0
    assert "usage" in data


def test_batch_embedding(mock_embedding_client):
    """Test batch embedding generation."""
    mock_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1] * 1024, "index": 0},
            {"object": "embedding", "embedding": [0.2] * 1024, "index": 1},
            {"object": "embedding", "embedding": [0.3] * 1024, "index": 2}
        ],
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "usage": {"prompt_tokens": 15, "total_tokens": 15}
    }
    mock_embedding_client.embed = AsyncMock(return_value=mock_response)

    request_data = {
        "input": ["text1", "text2", "text3"],
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }

    response = client.post("/v1/embeddings", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 3
    assert all(len(item["embedding"]) == 1024 for item in data["data"])


def test_invalid_empty_input():
    """Test that empty input is rejected."""
    request_data = {
        "input": "",
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }

    response = client.post("/v1/embeddings", json=request_data)
    assert response.status_code == 422  # Validation error


def test_invalid_empty_list():
    """Test that empty list input is rejected."""
    request_data = {
        "input": [],
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }

    response = client.post("/v1/embeddings", json=request_data)
    assert response.status_code == 422  # Validation error


def test_missing_required_fields():
    """Test that missing required fields are rejected."""
    response = client.post("/v1/embeddings", json={})
    assert response.status_code == 422


def test_service_unavailable(mock_embedding_client):
    """Test handling of vLLM service unavailability."""
    from src.utils import VLLMConnectionError

    mock_embedding_client.embed = AsyncMock(
        side_effect=VLLMConnectionError("Service unavailable")
    )

    request_data = {
        "input": "test text",
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }

    response = client.post("/v1/embeddings", json=request_data)
    assert response.status_code == 503
    assert "detail" in response.json()
    assert "error" in response.json()["detail"]


def test_encoding_format_parameter(mock_embedding_client, mock_vllm_embedding_response):
    """Test encoding_format parameter."""
    mock_embedding_client.embed = AsyncMock(return_value=mock_vllm_embedding_response)

    request_data = {
        "input": "test",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "encoding_format": "float"
    }

    response = client.post("/v1/embeddings", json=request_data)
    assert response.status_code == 200
