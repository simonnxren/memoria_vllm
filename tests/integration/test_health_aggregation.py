"""Integration tests for health aggregation (requires running vLLM service)."""

import pytest
import httpx

BASE_URL = "http://localhost:8200"


@pytest.mark.integration
async def test_health_endpoint_structure():
    """Test health endpoint returns correct structure."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health", timeout=10.0)

        # May be 200 or 503 depending on service state
        assert response.status_code in [200, 503]

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "router" in data["services"]
        assert "embedding" in data["services"]
        assert "generation" in data["services"]
        assert "timestamp" in data
        assert "uptime_seconds" in data


@pytest.mark.integration
async def test_health_uptime_increases():
    """Test that uptime increases over time."""
    async with httpx.AsyncClient() as client:
        response1 = await client.get(f"{BASE_URL}/health", timeout=10.0)
        data1 = response1.json()
        uptime1 = data1["uptime_seconds"]

        # Wait a moment
        import asyncio
        await asyncio.sleep(1)

        response2 = await client.get(f"{BASE_URL}/health", timeout=10.0)
        data2 = response2.json()
        uptime2 = data2["uptime_seconds"]

        assert uptime2 > uptime1
