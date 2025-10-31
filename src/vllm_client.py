"""HTTP client for communicating with vLLM instances."""

import httpx
import logging
from typing import Dict, Any, List, Union
from src.settings import settings
from src.utils import VLLMConnectionError, InvalidRequestError

logger = logging.getLogger(__name__)


class VLLMClient:
    """Base client for vLLM HTTP communication."""

    def __init__(self, base_url: str, timeout: int = None):
        """Initialize vLLM client.

        Args:
            base_url: Base URL of the vLLM instance
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout or settings.http_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to vLLM instance.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for httpx request

        Returns:
            Response JSON data

        Raises:
            VLLMConnectionError: If unable to connect to vLLM
            InvalidRequestError: If vLLM returns 4xx error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                error_detail = e.response.text
                logger.error(f"vLLM bad request: {error_detail}")
                raise InvalidRequestError(f"Invalid request to vLLM: {error_detail}")
            else:
                logger.error(f"vLLM server error: {e.response.status_code}")
                raise VLLMConnectionError(f"vLLM server error: {e.response.status_code}")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.error(f"Failed to connect to vLLM at {url}: {str(e)}")
            raise VLLMConnectionError(f"Unable to connect to vLLM service at {self.base_url}")

    async def health_check(self) -> bool:
        """Check if vLLM instance is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=settings.health_check_timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {str(e)}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class VLLMEmbeddingClient(VLLMClient):
    """Client for vLLM embedding instance."""

    def __init__(self):
        super().__init__(settings.vllm_embed_url)

    async def embed(self, texts: Union[str, List[str]], model: str) -> Dict[str, Any]:
        """Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            model: Model name

        Returns:
            Embedding response from vLLM
        """
        payload = {
            "input": texts,
            "model": model
        }

        logger.info(f"Requesting embeddings for {len(texts) if isinstance(texts, list) else 1} text(s)")
        response = await self._request("POST", "/v1/embeddings", json=payload)
        logger.info(f"Successfully generated {len(response.get('data', []))} embeddings")

        return response


class VLLMGenerationClient(VLLMClient):
    """Client for vLLM generation instance."""

    def __init__(self):
        super().__init__(settings.vllm_gen_url)

    async def complete(self, prompt: Union[str, List[str]], **params) -> Dict[str, Any]:
        """Generate text completion.

        Args:
            prompt: Prompt text or list of prompts
            **params: Additional completion parameters (max_tokens, temperature, etc.)

        Returns:
            Completion response from vLLM
        """
        payload = {
            "prompt": prompt,
            **params
        }

        logger.info(f"Requesting completion for prompt length: {len(prompt) if isinstance(prompt, str) else len(prompt)}")
        response = await self._request("POST", "/v1/completions", json=payload)
        logger.info(f"Successfully generated completion with {len(response.get('choices', []))} choices")

        return response
