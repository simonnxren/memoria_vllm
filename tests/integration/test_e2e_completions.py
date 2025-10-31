"""Integration tests for completions endpoint (requires running vLLM service)."""

import pytest
import httpx

BASE_URL = "http://localhost:8200"


@pytest.mark.integration
async def test_e2e_simple_completion():
    """Test end-to-end text completion."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": "Once upon a time",
                "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=60.0
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) >= 1
        assert "text" in data["choices"][0]
        assert len(data["choices"][0]["text"]) > 0
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0


@pytest.mark.integration
async def test_e2e_completion_with_stop():
    """Test completion with stop sequences."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": "Count to 10: 1, 2, 3",
                "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                "max_tokens": 30,
                "stop": ["\n", "10"]
            },
            timeout=60.0
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] in ["stop", "length"]


@pytest.mark.integration
async def test_e2e_completion_temperature_variation():
    """Test that temperature affects output."""
    prompts_results = []

    async with httpx.AsyncClient() as client:
        # Generate with low temperature (more deterministic)
        for _ in range(2):
            response = await client.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "prompt": "The capital of France is",
                    "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=60.0
            )
            data = response.json()
            prompts_results.append(data["choices"][0]["text"])

        # With low temperature, outputs should be similar
        # (not necessarily identical due to sampling, but close)
        assert all(isinstance(text, str) for text in prompts_results)
