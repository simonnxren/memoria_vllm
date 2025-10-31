"""Integration tests for embeddings endpoint (requires running vLLM service)."""

import pytest
import httpx

BASE_URL = "http://localhost:8200"


@pytest.mark.integration
async def test_e2e_single_embedding():
    """Test end-to-end single text embedding."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={
                "input": "Hello world",
                "model": "Qwen/Qwen3-Embedding-0.6B"
            },
            timeout=30.0
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) > 0  # Actual embedding dimension
        assert "usage" in data


@pytest.mark.integration
async def test_e2e_batch_embeddings():
    """Test end-to-end batch embedding."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Python is a great programming language"
    ]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={
                "input": texts,
                "model": "Qwen/Qwen3-Embedding-0.6B"
            },
            timeout=30.0
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert all("embedding" in item for item in data["data"])
        assert all(item["index"] == i for i, item in enumerate(data["data"]))


@pytest.mark.integration
async def test_e2e_embedding_dimensions_consistent():
    """Test that all embeddings have consistent dimensions."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={
                "input": ["text1", "text2"],
                "model": "Qwen/Qwen3-Embedding-0.6B"
            },
            timeout=30.0
        )

        data = response.json()
        dim1 = len(data["data"][0]["embedding"])
        dim2 = len(data["data"][1]["embedding"])
        assert dim1 == dim2
