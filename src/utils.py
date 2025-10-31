"""Utilities for error handling and logging."""

import logging
import sys
from typing import Optional, Dict, Any
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from src.settings import settings


# ============================================================================
# ERROR HANDLING
# ============================================================================

class OpenAIError(BaseModel):
    """OpenAI-compatible error response format."""
    message: str
    type: str
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response wrapper."""
    error: OpenAIError


class VLLMServiceError(Exception):
    """Base exception for vLLM service errors."""
    def __init__(self, message: str, status_code: int = 500, error_type: str = "internal_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

    def to_openai_error(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible error format."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.error_type
            }
        }


class VLLMConnectionError(VLLMServiceError):
    """Raised when unable to connect to vLLM instance."""
    def __init__(self, message: str = "vLLM service unavailable"):
        super().__init__(message, status_code=503, error_type="service_unavailable")


class InvalidRequestError(VLLMServiceError):
    """Raised for invalid request parameters."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="invalid_request_error")


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configure logging based on settings."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_format.lower() == "json":
        # JSON formatter
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Initialize logger
logger = setup_logging()
