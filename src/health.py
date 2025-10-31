"""Health check endpoint implementation."""

from fastapi import APIRouter, Response
from src.models import HealthResponse
from src.vllm_client import VLLMEmbeddingClient, VLLMGenerationClient
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start time
service_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    """Aggregate health check for all services.

    Checks the health of both vLLM instances (embedding and generation)
    and returns an aggregated status.

    Returns:
        HealthResponse with status of all services

    HTTP Status:
        200: All services healthy
        503: One or more services unhealthy (degraded state)
    """
    logger.debug("Health check requested")

    # Initialize clients for health checks
    embedding_client = VLLMEmbeddingClient()
    generation_client = VLLMGenerationClient()

    # Check embedding service
    embedding_healthy = await embedding_client.health_check()

    # Check generation service
    generation_healthy = await generation_client.health_check()

    # Close clients
    await embedding_client.close()
    await generation_client.close()

    # Determine overall status
    all_healthy = embedding_healthy and generation_healthy

    if not all_healthy:
        response.status_code = 503
        logger.warning(
            f"Health check degraded - Embedding: {embedding_healthy}, "
            f"Generation: {generation_healthy}"
        )

    # Calculate uptime
    uptime = time.time() - service_start_time

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services={
            "router": "healthy",
            "embedding": "healthy" if embedding_healthy else "unhealthy",
            "generation": "healthy" if generation_healthy else "unhealthy"
        },
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )
