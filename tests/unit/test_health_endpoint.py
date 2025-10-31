"""Unit tests for health endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_clients():
    """Mock both vLLM clients."""
    with patch("src.health.VLLMEmbeddingClient") as mock_embed, \
         patch("src.health.VLLMGenerationClient") as mock_gen:
        yield mock_embed, mock_gen


def test_health_all_services_healthy(mock_clients):
    """Test health check when all services are healthy."""
    mock_embed, mock_gen = mock_clients

    # Configure mocks
    embed_instance = mock_embed.return_value
    gen_instance = mock_gen.return_value
    embed_instance.health_check = AsyncMock(return_value=True)
    gen_instance.health_check = AsyncMock(return_value=True)
    embed_instance.close = AsyncMock()
    gen_instance.close = AsyncMock()

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["router"] == "healthy"
    assert data["services"]["embedding"] == "healthy"
    assert data["services"]["generation"] == "healthy"
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_health_embedding_unhealthy(mock_clients):
    """Test health check when embedding service is down."""
    mock_embed, mock_gen = mock_clients

    embed_instance = mock_embed.return_value
    gen_instance = mock_gen.return_value
    embed_instance.health_check = AsyncMock(return_value=False)
    gen_instance.health_check = AsyncMock(return_value=True)
    embed_instance.close = AsyncMock()
    gen_instance.close = AsyncMock()

    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["embedding"] == "unhealthy"
    assert data["services"]["generation"] == "healthy"


def test_health_generation_unhealthy(mock_clients):
    """Test health check when generation service is down."""
    mock_embed, mock_gen = mock_clients

    embed_instance = mock_embed.return_value
    gen_instance = mock_gen.return_value
    embed_instance.health_check = AsyncMock(return_value=True)
    gen_instance.health_check = AsyncMock(return_value=False)
    embed_instance.close = AsyncMock()
    gen_instance.close = AsyncMock()

    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["embedding"] == "healthy"
    assert data["services"]["generation"] == "unhealthy"


def test_health_all_services_unhealthy(mock_clients):
    """Test health check when all services are down."""
    mock_embed, mock_gen = mock_clients

    embed_instance = mock_embed.return_value
    gen_instance = mock_gen.return_value
    embed_instance.health_check = AsyncMock(return_value=False)
    gen_instance.health_check = AsyncMock(return_value=False)
    embed_instance.close = AsyncMock()
    gen_instance.close = AsyncMock()

    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["embedding"] == "unhealthy"
    assert data["services"]["generation"] == "unhealthy"
    assert data["services"]["router"] == "healthy"  # Router itself is healthy
