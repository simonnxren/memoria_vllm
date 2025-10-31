"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model Configuration
    model_embed_path: str = "/media/simon/Data/models/hf_models/Qwen3-Embedding-0.6B"
    model_gen_path: str = "/media/simon/Data/models/hf_models/Qwen3-8B-FP8"

    # Service Ports
    router_port: int = 8200
    vllm_embed_port: int = 8100
    vllm_gen_port: int = 8101

    # vLLM V1 Configuration (V1 is default since v0.8.0+)
    vllm_embed_gpu_memory: float = 0.3
    vllm_gen_gpu_memory: float = 0.6

    # vLLM Performance Tuning - Embedding Instance
    vllm_embed_max_num_seqs: int = 256  # Max concurrent sequences for embeddings
    vllm_embed_max_batched_tokens: int = 8192  # Max tokens per batch for embeddings
    vllm_embed_pooler_config: str = '{"pooling_type":"MEAN","normalize":true,"enable_chunked_processing":true,"max_embed_len":8192}'

    # vLLM Performance Tuning - Generation Instance
    vllm_gen_max_num_seqs: int = 128  # Max concurrent sequences for generation
    vllm_gen_max_batched_tokens: int = 16384  # Max tokens per batch for generation
    vllm_gen_max_model_len: int = 8192  # Maximum context length
    vllm_gen_scheduler_policy: Literal["fcfs", "priority"] = "fcfs"  # First-Come-First-Serve or priority-based

    # vLLM Instance URLs
    vllm_embed_url: str = "http://vllm-embedding:8100"
    vllm_gen_url: str = "http://vllm-generation:8101"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Model Names (as returned in API responses)
    model_embed_name: str = "qwen3-embedding-0.6b"
    model_gen_name: str = "qwen3-8b-fp8"

    # HTTP Client Settings
    http_timeout: int = 30
    health_check_timeout: int = 5

    # Router Configuration
    enable_metrics: bool = False  # Enable Prometheus metrics
    enable_request_logging: bool = False  # Enable detailed request logging
    max_log_len: Optional[int] = None  # Max chars to log (None = unlimited)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
