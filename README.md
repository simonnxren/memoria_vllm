# memoria_vllm

Production-ready vLLM inference service with OpenAI-compatible API for embeddings and text generation.

## Features

- 🚀 **OpenAI-Compatible API** - Drop-in replacement for OpenAI embeddings and completions endpoints
- ⚡ **vLLM V1 Engine** - Latest version with 1.7x speedup and optimized performance
- 🐳 **Docker Compose** - Easy deployment with isolated embedding and generation instances
- 🔧 **GPU Memory Management** - Configurable memory allocation per model
- 📊 **Health Monitoring** - Built-in health checks for all services
- ✅ **Fully Tested** - 64 tests covering unit, integration, compliance, and load scenarios

---

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env to configure your models
# Supports HuggingFace model identifiers or local paths
```

**Key Configuration:**
```bash
MODEL_EMBED_NAME=Qwen/Qwen3-Embedding-0.6B    # HuggingFace model ID
MODEL_GEN_NAME=Qwen/Qwen3-4B-Thinking-2507-FP8
VLLM_EMBED_GPU_MEMORY=0.3                      # 30% GPU memory
VLLM_GEN_GPU_MEMORY=0.6                        # 60% GPU memory
HTTP_TIMEOUT=120                                # Request timeout (seconds)
```

### 3. Deploy

```bash
# Start all services
sudo docker compose up -d

# Check health
curl http://localhost:8200/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "router": "healthy",
    "embedding": "healthy", 
    "generation": "healthy"
  }
}
```

---

## API Integration

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Generate text embeddings |
| `/v1/completions` | POST | Generate text completions |
| `/health` | GET | Service health check |

### Base URL
```
http://localhost:8200
```

### Python Integration

#### Using OpenAI SDK

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed"  # API key not required
)

# Generate embeddings
response = client.embeddings.create(
    input="Your text here",
    model="Qwen/Qwen3-Embedding-0.6B"
)
embedding = response.data[0].embedding  # List of 1024 floats

# Generate completions
response = client.completions.create(
    prompt="Once upon a time",
    model="Qwen/Qwen3-4B-Thinking-2507-FP8",
    max_tokens=100,
    temperature=0.7
)
text = response.choices[0].text
```

#### Using httpx/requests

```python
import httpx

BASE_URL = "http://localhost:8200/v1"

# Embeddings
response = httpx.post(
    f"{BASE_URL}/embeddings",
    json={
        "input": "Your text here",
        "model": "Qwen/Qwen3-Embedding-0.6B"
    }
)
data = response.json()
embedding = data["data"][0]["embedding"]

# Completions
response = httpx.post(
    f"{BASE_URL}/completions",
    json={
        "prompt": "Once upon a time",
        "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "max_tokens": 100
    }
)
data = response.json()
text = data["choices"][0]["text"]
```

### cURL Examples

```bash
# Embeddings
curl -X POST http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'

# Completions
curl -X POST http://localhost:8200/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
    "max_tokens": 50
  }'

# Health check
curl http://localhost:8200/health
```

### JavaScript/TypeScript Integration

```typescript
// Using fetch API
async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch('http://localhost:8200/v1/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: text,
      model: 'Qwen/Qwen3-Embedding-0.6B'
    })
  });
  
  const data = await response.json();
  return data.data[0].embedding;
}

async function getCompletion(prompt: string): Promise<string> {
  const response = await fetch('http://localhost:8200/v1/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      model: 'Qwen/Qwen3-4B-Thinking-2507-FP8',
      max_tokens: 100
    })
  });
  
  const data = await response.json();
  return data.choices[0].text;
}
```

---

## API Reference

### POST /v1/embeddings

**Request:**
```json
{
  "input": "string or array of strings",
  "model": "model_identifier",
  "encoding_format": "float"  // optional
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],  // 1024-dim vector
      "index": 0
    }
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### POST /v1/completions

**Request:**
```json
{
  "prompt": "string",
  "model": "model_identifier",
  "max_tokens": 100,           // optional, default: 16
  "temperature": 0.7,          // optional, 0.0-2.0
  "top_p": 0.9,               // optional, 0.0-1.0
  "n": 1,                     // optional, number of completions
  "stop": [".", "\n"],        // optional, stop sequences
  "presence_penalty": 0.0,    // optional, -2.0 to 2.0
  "frequency_penalty": 0.0    // optional, -2.0 to 2.0
}
```

**Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
  "choices": [
    {
      "text": "generated text",
      "index": 0,
      "finish_reason": "length"  // or "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Client Application                  │
└────────────────────┬────────────────────────────┘
                     │ HTTP
                     ▼
         ┌───────────────────────┐
         │   FastAPI Router      │
         │   (Port 8200)         │
         └─────────┬─────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌────────────────┐   ┌────────────────┐
│ vLLM Embedding │   │ vLLM Generation│
│  (Port 8100)   │   │  (Port 8101)   │
│                │   │                │
│ GPU Memory 30% │   │ GPU Memory 60% │
└────────────────┘   └────────────────┘
```

**Components:**
- **Router Service**: FastAPI gateway for request routing and health aggregation
- **Embedding Instance**: Dedicated vLLM instance for embedding generation
- **Generation Instance**: Dedicated vLLM instance for text completion

---

## Development

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Python environment
cp .env.example .env
```

### Testing

```bash
# Run all tests (64 tests)
pytest

# Run specific test suites
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/compliance/        # OpenAI compatibility
pytest tests/load/              # Performance benchmarks
pytest tests/examples/          # Usage examples

# Run with coverage
pytest --cov=src --cov-report=html
```

### Docker Management

```bash
# Start services
sudo docker compose up -d

# View logs
sudo docker compose logs -f
sudo docker compose logs vllm-router
sudo docker compose logs vllm-embedding
sudo docker compose logs vllm-generation

# Restart specific service
sudo docker restart vllm-router

# Stop all services
sudo docker compose down

# Stop and remove volumes
sudo docker compose down -v
```

### Check Individual Services

```bash
# Embedding service health
curl http://localhost:8100/health

# Generation service health  
curl http://localhost:8101/health

# Router aggregated health
curl http://localhost:8200/health
```

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_EMBED_NAME` | HuggingFace model ID for embeddings | - |
| `MODEL_GEN_NAME` | HuggingFace model ID for generation | - |
| `ROUTER_PORT` | Router service port | 8200 |
| `VLLM_EMBED_PORT` | Embedding instance port | 8100 |
| `VLLM_GEN_PORT` | Generation instance port | 8101 |
| `VLLM_EMBED_GPU_MEMORY` | GPU memory fraction for embedding | 0.3 |
| `VLLM_GEN_GPU_MEMORY` | GPU memory fraction for generation | 0.6 |
| `VLLM_EMBED_MAX_NUM_SEQS` | Max concurrent sequences (embedding) | 256 |
| `VLLM_GEN_MAX_NUM_SEQS` | Max concurrent sequences (generation) | 128 |
| `HTTP_TIMEOUT` | Request timeout in seconds | 120 |
| `LOG_LEVEL` | Logging level | INFO |

### GPU Memory Allocation

Total GPU memory usage should not exceed 95%:
- Embedding: 30% (0.3)
- Generation: 60% (0.6)
- System overhead: ~10%

Adjust `VLLM_EMBED_GPU_MEMORY` and `VLLM_GEN_GPU_MEMORY` based on your models and GPU capacity.

---

## Performance

### vLLM V1 Engine Benefits

- **1.7x faster** inference compared to legacy engine
- Optimized CPU-GPU overlap
- Unified scheduler (no prefill/decode distinction)
- Automatic chunked prefill for long contexts
- Zero-overhead prefix caching

### Recommended Settings

**Embedding Model (Batch Processing):**
- `VLLM_EMBED_MAX_NUM_SEQS=256`
- `VLLM_EMBED_MAX_BATCHED_TOKENS=8192`

**Generation Model (Interactive):**
- `VLLM_GEN_MAX_NUM_SEQS=128`
- `VLLM_GEN_MAX_BATCHED_TOKENS=16384`
- `VLLM_GEN_MAX_MODEL_LEN=8192`

---

## Troubleshooting

### Service Won't Start

```bash
# Check Docker logs
sudo docker compose logs

# Check GPU availability
nvidia-smi

# Verify Docker has GPU access
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model Not Found

Ensure model names in `.env` match HuggingFace repository names:
```bash
MODEL_EMBED_NAME=Qwen/Qwen3-Embedding-0.6B  # Correct format
```

### Timeout Errors

Increase timeout for large models:
```bash
HTTP_TIMEOUT=300  # 5 minutes
```

### Out of Memory

Reduce GPU memory allocation:
```bash
VLLM_EMBED_GPU_MEMORY=0.2
VLLM_GEN_GPU_MEMORY=0.5
```

---

## Project Structure

```
memoria_vllm/
├── src/
│   ├── main.py              # FastAPI application
│   ├── embeddings.py        # Embedding endpoint
│   ├── completions.py       # Completion endpoint
│   ├── health.py            # Health check endpoint
│   ├── models.py            # Pydantic request/response models
│   ├── settings.py          # Configuration management
│   ├── utils.py             # Error handling & logging
│   └── vllm_client.py       # vLLM HTTP client
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── compliance/          # OpenAI compatibility tests
│   ├── load/                # Performance benchmarks
│   └── examples/            # Usage examples
├── docker/
│   └── Dockerfile.router    # Router service Dockerfile
├── docker-compose.yml       # Service orchestration
├── .env.example             # Configuration template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## License

MIT License - see LICENSE file for details

---

## Additional Documentation

- [VLLM_SERVICE_PLAN.md](VLLM_SERVICE_PLAN.md) - Detailed architecture and development plan
- [CLAUDE.md](CLAUDE.md) - Development guidelines for AI assistants

---

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker logs: `sudo docker compose logs`
3. Verify configuration in `.env`
4. Test individual services at ports 8100, 8101, 8200
