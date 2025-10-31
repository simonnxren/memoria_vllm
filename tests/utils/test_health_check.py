"""Health check utility tests."""

import pytest
import httpx


BASE_URL = "http://localhost:8200"


@pytest.mark.integration
async def test_router_health():
    """Test router health endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "embedding" in data["services"]
        assert "generation" in data["services"]


@pytest.mark.integration
async def test_embedding_endpoint_functional():
    """Test that embedding endpoint is working."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={
                "input": "test",
                "model": "Qwen/Qwen3-Embedding-0.6B"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0


@pytest.mark.integration
async def test_completion_endpoint_functional():
    """Test that completion endpoint is working."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": "test",
                "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                "max_tokens": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
