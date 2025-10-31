"""Completions endpoint implementation."""

from fastapi import APIRouter, HTTPException
from src.models import CompletionRequest, CompletionResponse
from src.vllm_client import VLLMGenerationClient
from src.utils import VLLMConnectionError, InvalidRequestError
from src.settings import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize generation client
generation_client = VLLMGenerationClient()


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Generate text completion for given prompt.

    This endpoint is OpenAI-compatible and supports various completion
    parameters like temperature, max_tokens, top_p, etc.

    Args:
        request: Completion request with prompt and parameters

    Returns:
        CompletionResponse with generated text

    Raises:
        HTTPException: 400 for invalid requests, 503 for service unavailable
    """
    try:
        logger.info(f"Completion request received for model: {request.model}")

        # Prepare parameters for vLLM
        params = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
        }

        # Add optional parameters if provided
        if request.stop is not None:
            params["stop"] = request.stop
        if request.presence_penalty != 0.0:
            params["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty != 0.0:
            params["frequency_penalty"] = request.frequency_penalty

        # Forward request to vLLM generation instance
        response = await generation_client.complete(
            prompt=request.prompt,
            **params
        )

        # Ensure response uses configured model name
        response["model"] = settings.model_gen_name

        return response

    except InvalidRequestError as e:
        logger.error(f"Invalid completion request: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_openai_error()
        )
    except VLLMConnectionError as e:
        logger.error(f"Generation service unavailable: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_openai_error()
        )
    except Exception as e:
        logger.exception(f"Unexpected error in completion endpoint: {str(e)}")
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
