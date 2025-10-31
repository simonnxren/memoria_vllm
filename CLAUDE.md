# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **standalone vLLM inference service** with an OpenAI-compatible API. The service is designed as a reusable microservice that can be deployed once and integrated with any application requiring fast, production-grade LLM inference (RAG systems, chatbots, AI applications, etc.).

**Current Status**: Planning phase - implementation not yet started (see VLLM_SERVICE_PLAN.md)

## Architecture

The service consists of three main components:

1. **FastAPI Router** (Port 8200)
   - API gateway that routes requests to vLLM instances
   - Implements OpenAI-compatible endpoints: `/v1/embeddings`, `/v1/completions`, `/health`
   - Aggregates health checks from both vLLM instances

2. **vLLM Embedding Instance** (Port 8100)
   - Dedicated embedding model server
   - Model: Qwen3-Embedding-0.6B (configurable)
   - Handles vector embedding generation

3. **vLLM Generation Instance** (Port 8101)
   - Dedicated text generation model server
   - Model: Qwen3-8B-FP8 (configurable)
   - Handles text completions

**Design Philosophy**: Dual vLLM instances prevent model switching overhead and enable concurrent processing. All services are stateless (pure request/response pattern).

## Configuration

Key environment variables (defined in `.env`):
- Model paths (local filesystem)
- Service ports (Router: 8200, Embedding: 8100, Generation: 8101)
- GPU memory allocation
- Logging configuration

## API Specification

### POST /v1/embeddings
OpenAI-compatible embedding generation. Accepts single text string or batch of texts.

### POST /v1/completions
OpenAI-compatible text completion. Supports temperature, max_tokens, top_p, stop sequences.

### GET /health
Aggregates health status from both vLLM instances. Returns 200 if all healthy, 503 if degraded.

## Implementation Guidelines

### Request/Response Handling
- All Pydantic schemas must match OpenAI API specification exactly
- Use `src/models/requests.py` for request models
- Use `src/models/responses.py` for response models
- Validation errors should return 400 with OpenAI-compatible error format

### Error Handling
Error responses must match OpenAI format with appropriate status codes:
- 200: Success
- 400: Invalid request (validation errors)
- 503: Service unavailable (vLLM instance down)
- 500: Internal server error

### vLLM Client Communication
- Use async HTTP requests to vLLM instances
- Implement retry logic with exponential backoff
- Set appropriate timeouts
- Handle connection errors gracefully

### Testing Best Practices
- **Unit tests**: Mock all vLLM HTTP responses, should run in <1s
- **Integration tests**: Require running Docker services, mark with `@pytest.mark.integration`
- **Compliance tests**: Use official `openai` Python SDK to verify compatibility
- **Load tests**: Measure concurrent request handling (50+ simultaneous requests)

### Performance Targets
- Embedding throughput: 100+ embed/sec
- Embedding latency (p95): < 500ms
- Completion latency (p95): < 5s for 100 tokens
- Concurrent users: 50+ simultaneous requests
- Memory: No leaks during 24-hour runs

## Common Troubleshooting

### vLLM instances won't start
- Check model paths exist
- Verify GPU availability
- Check Docker logs
- Reduce GPU memory allocation if OOM errors occur

### Health check returns 503
- Test individual instances
- Check if containers are running
- Restart specific service

### Slow performance
- Monitor GPU utilization
- Use batch requests for embeddings
- Check vLLM logs for warnings

## Development Phases

The project follows an 8-phase implementation plan (see VLLM_SERVICE_PLAN.md):
1. Project bootstrap (Day 1)
2. vLLM infrastructure setup (Day 2)
3. FastAPI router - embeddings endpoint (Day 3)
4. FastAPI router - completions endpoint (Day 4)
5. Health monitoring and observability (Day 5)
6. OpenAI compliance testing (Days 6-7)
7. Performance benchmarking (Day 8)
8. Documentation (Day 9)
