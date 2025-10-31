"""Example client usage tests demonstrating API usage patterns."""

import pytest
from openai import OpenAI


@pytest.fixture
def client():
    """OpenAI client configured for local vLLM service."""
    return OpenAI(
        base_url="http://localhost:8200/v1",
        api_key="not-needed"
    )


@pytest.mark.integration
def test_single_embedding(client):
    """Example: Generate embedding for single text."""
    response = client.embeddings.create(
        input="Hello, world!",
        model="Qwen/Qwen3-Embedding-0.6B"
    )
    
    embedding = response.data[0].embedding
    assert len(embedding) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.integration
def test_batch_embeddings(client):
    """Example: Generate embeddings for multiple texts."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Python is a versatile programming language"
    ]
    
    response = client.embeddings.create(
        input=texts,
        model="Qwen/Qwen3-Embedding-0.6B"
    )
    
    assert len(response.data) == len(texts)
    for item in response.data:
        assert len(item.embedding) > 0


@pytest.mark.integration
def test_text_completion(client):
    """Example: Generate text completion."""
    prompt = "Once upon a time in a land far away"
    
    response = client.completions.create(
        prompt=prompt,
        model="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=100,
        temperature=0.7
    )
    
    completion = response.choices[0].text
    assert len(completion) > 0
    assert response.usage.total_tokens > 0
    assert response.choices[0].finish_reason in ["length", "stop"]


@pytest.mark.integration
def test_completion_with_parameters(client):
    """Example: Text completion with various parameters."""
    response = client.completions.create(
        prompt="List three benefits of exercise:",
        model="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=150,
        temperature=0.3,
        top_p=0.9,
        stop=["\n\n"]
    )
    
    assert len(response.choices[0].text) > 0
    assert response.choices[0].finish_reason in ["length", "stop"]


@pytest.mark.integration
def test_rag_workflow(client):
    """Example: RAG (Retrieval-Augmented Generation) workflow."""
    # Step 1: Embed a query
    query = "What are the benefits of renewable energy?"
    query_embedding_response = client.embeddings.create(
        input=query,
        model="Qwen/Qwen3-Embedding-0.6B"
    )
    query_embedding = query_embedding_response.data[0].embedding
    assert len(query_embedding) > 0
    
    # Step 2: Simulate retrieval
    retrieved_context = """
    Renewable energy sources like solar and wind power offer several benefits:
    1. Reduced greenhouse gas emissions
    2. Lower long-term costs
    3. Energy independence
    4. Sustainable and inexhaustible
    """
    
    # Step 3: Generate answer using retrieved context
    prompt = f"""Context: {retrieved_context}

Question: {query}

Answer:"""
    
    response = client.completions.create(
        prompt=prompt,
        model="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=150,
        temperature=0.5
    )
    
    answer = response.choices[0].text.strip()
    assert len(answer) > 0
