"""Embeddings endpoint implementation."""

from fastapi import APIRouter, HTTPException
from src.models import EmbeddingRequest, EmbeddingResponse
from src.vllm_client import VLLMEmbeddingClient
from src.utils import VLLMConnectionError, InvalidRequestError
from src.settings import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize embedding client
embedding_client = VLLMEmbeddingClient()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Generate embeddings for input text(s).

    This endpoint is OpenAI-compatible and supports both single text
    and batch embedding requests.

    Args:
        request: Embedding request with input text(s)

    Returns:
        EmbeddingResponse with generated embeddings

    Raises:
        HTTPException: 400 for invalid requests, 503 for service unavailable
    """
    try:
        logger.info(f"Embedding request received for model: {request.model}")

        # Forward request to vLLM embedding instance
        response = await embedding_client.embed(
            texts=request.input,
            model=request.model
        )

        # Ensure response uses configured model name
        response["model"] = settings.model_embed_name

        return response

    except InvalidRequestError as e:
        logger.error(f"Invalid embedding request: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_openai_error()
        )
    except VLLMConnectionError as e:
        logger.error(f"Embedding service unavailable: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_openai_error()
        )
    except Exception as e:
        logger.exception(f"Unexpected error in embedding endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )
