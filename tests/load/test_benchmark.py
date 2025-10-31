"""Performance benchmarking tests for vLLM service."""

import asyncio
import pytest
import httpx
import time
import statistics
from typing import List


BASE_URL = "http://localhost:8200"


@pytest.mark.integration
async def test_embedding_throughput():
    """Benchmark embedding throughput and latency (single requests)."""
    num_requests = 100
    latencies = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(num_requests):
            start = time.time()
            response = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={
                    "input": f"Sample text number {i}",
                    "model": "Qwen/Qwen3-Embedding-0.6B"
                }
            )
            response.raise_for_status()
            latencies.append(time.time() - start)

    total_time = sum(latencies)
    print(f"\nEmbedding Benchmark:")
    print(f"  Requests: {num_requests}")
    print(f"  Mean latency: {statistics.mean(latencies):.3f}s")
    print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")
    print(f"  Throughput: {num_requests / total_time:.1f} req/sec")
    
    # Performance assertions
    assert statistics.mean(latencies) < 1.0, "Mean latency too high"
    assert statistics.quantiles(latencies, n=20)[18] < 2.0, "P95 latency too high"


@pytest.mark.integration
async def test_batch_embedding_throughput():
    """Benchmark batch embedding throughput."""
    num_requests = 50
    batch_size = 10
    latencies = []
    texts = [f"Sample text {i}" for i in range(batch_size)]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(num_requests):
            start = time.time()
            response = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={
                    "input": texts,
                    "model": "Qwen/Qwen3-Embedding-0.6B"
                }
            )
            response.raise_for_status()
            latencies.append(time.time() - start)

    total_embeddings = num_requests * batch_size
    total_time = sum(latencies)
    
    print(f"\nBatch Embedding Benchmark (batch_size={batch_size}):")
    print(f"  Total embeddings: {total_embeddings}")
    print(f"  Mean latency: {statistics.mean(latencies):.3f}s")
    print(f"  Throughput: {total_embeddings / total_time:.1f} embeddings/sec")
    
    assert statistics.mean(latencies) < 2.0, "Batch mean latency too high"


@pytest.mark.integration
async def test_completion_latency():
    """Benchmark completion latency and token throughput."""
    num_requests = 50
    max_tokens = 100
    latencies = []
    tokens_generated = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(num_requests):
            start = time.time()
            response = await client.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "prompt": "Once upon a time in a land far away",
                    "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            latencies.append(time.time() - start)
            
            data = response.json()
            tokens_generated.append(data["usage"]["completion_tokens"])

    total_tokens = sum(tokens_generated)
    total_time = sum(latencies)

    print(f"\nCompletion Benchmark (max_tokens={max_tokens}):")
    print(f"  Requests: {num_requests}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Mean latency: {statistics.mean(latencies):.3f}s")
    print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")
    print(f"  Throughput: {total_tokens / total_time:.1f} tokens/sec")
    
    assert statistics.mean(latencies) < 10.0, "Completion mean latency too high"
    assert total_tokens > 0, "No tokens generated"


@pytest.mark.integration
async def test_concurrent_requests():
    """Benchmark concurrent request handling."""
    num_concurrent = 50

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            client.post(
                f"{BASE_URL}/v1/embeddings",
                json={
                    "input": f"Concurrent test text {i}",
                    "model": "Qwen/Qwen3-Embedding-0.6B"
                }
            )
            for i in range(num_concurrent)
        ]

        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        successful = sum(
            1 for r in responses 
            if isinstance(r, httpx.Response) and r.status_code == 200
        )

        print(f"\nConcurrent Requests Benchmark:")
        print(f"  Concurrent requests: {num_concurrent}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {num_concurrent - successful}")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Requests/sec: {num_concurrent / duration:.1f}")
        
        # At least 80% should succeed
        assert successful >= num_concurrent * 0.8, "Too many failed concurrent requests"
        assert duration < 30.0, "Concurrent requests took too long"


@pytest.mark.integration
async def test_concurrent_both_models():
    """Test running embedding and completion models concurrently."""
    num_requests = 25  # 25 of each = 50 total concurrent requests

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create embedding requests
        embedding_tasks = [
            client.post(
                f"{BASE_URL}/v1/embeddings",
                json={
                    "input": f"Embedding test text {i}",
                    "model": "Qwen/Qwen3-Embedding-0.6B"
                }
            )
            for i in range(num_requests)
        ]
        
        # Create completion requests
        completion_tasks = [
            client.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "prompt": f"Complete this: Test number {i}",
                    "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                    "max_tokens": 20
                }
            )
            for i in range(num_requests)
        ]
        
        # Run both types concurrently
        all_tasks = embedding_tasks + completion_tasks
        
        start = time.time()
        responses = await asyncio.gather(*all_tasks, return_exceptions=True)
        duration = time.time() - start
        
        # Separate results by type
        embedding_responses = responses[:num_requests]
        completion_responses = responses[num_requests:]
        
        embedding_successful = sum(
            1 for r in embedding_responses
            if isinstance(r, httpx.Response) and r.status_code == 200
        )
        
        completion_successful = sum(
            1 for r in completion_responses
            if isinstance(r, httpx.Response) and r.status_code == 200
        )
        
        total_successful = embedding_successful + completion_successful
        total_requests = num_requests * 2
        
        print(f"\nConcurrent Both Models Benchmark:")
        print(f"  Embedding requests: {num_requests}")
        print(f"    Successful: {embedding_successful}")
        print(f"    Failed: {num_requests - embedding_successful}")
        print(f"  Completion requests: {num_requests}")
        print(f"    Successful: {completion_successful}")
        print(f"    Failed: {num_requests - completion_successful}")
        print(f"  Total successful: {total_successful}/{total_requests}")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Requests/sec: {total_requests / duration:.1f}")
        
        # At least 80% should succeed for each model
        assert embedding_successful >= num_requests * 0.8, f"Too many failed embedding requests: {num_requests - embedding_successful}/{num_requests}"
        assert completion_successful >= num_requests * 0.8, f"Too many failed completion requests: {num_requests - completion_successful}/{num_requests}"
        assert duration < 60.0, "Concurrent requests took too long"


@pytest.mark.integration
async def test_service_health():
    """Verify service is healthy before running benchmarks."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy", f"Service unhealthy: {data}"
