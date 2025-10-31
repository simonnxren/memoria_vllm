"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import Union, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# REQUEST MODELS
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Request model for /v1/embeddings endpoint."""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(..., description="Model identifier")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format for embeddings")
    user: Optional[str] = Field(default=None, description="User identifier")

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        """Ensure input is not empty."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if not all(isinstance(item, str) and item.strip() for item in v):
                raise ValueError("All input items must be non-empty strings")
        return v


class CompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint."""
    model: str = Field(..., description="Model identifier")
    prompt: Union[str, List[str]] = Field(..., description="Prompt(s) for completion")
    max_tokens: Optional[int] = Field(default=16, description="Maximum tokens to generate", ge=1)
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, description="Number of completions", ge=1, le=10)
    stream: Optional[bool] = Field(default=False, description="Stream responses")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    user: Optional[str] = Field(default=None, description="User identifier")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        """Ensure prompt is not empty."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Prompt cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Prompt list cannot be empty")
            if not all(isinstance(item, str) and item.strip() for item in v):
                raise ValueError("All prompt items must be non-empty strings")
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(default=0, description="Number of tokens in completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class EmbeddingData(BaseModel):
    """Single embedding data item."""
    object: str = Field(default="embedding", description="Object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index in the list")


class EmbeddingResponse(BaseModel):
    """Response model for /v1/embeddings endpoint."""
    object: str = Field(default="list", description="Object type")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model identifier")
    usage: UsageInfo = Field(..., description="Token usage information")


class CompletionChoice(BaseModel):
    """Single completion choice."""
    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    logprobs: Optional[Any] = Field(default=None, description="Log probabilities")
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion finish")


class CompletionResponse(BaseModel):
    """Response model for /v1/completions endpoint."""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model identifier")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = Field(..., description="Overall health status")
    services: dict = Field(..., description="Health status of individual services")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
