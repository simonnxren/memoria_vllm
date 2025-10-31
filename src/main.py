"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src import embeddings, completions, health
from src.utils import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="vLLM OpenAI Service",
    description="Production-ready vLLM inference service with OpenAI-compatible API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(completions.router, tags=["completions"])
app.include_router(health.router, tags=["health"])


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("vLLM OpenAI Service starting up")
    logger.info(f"Embedding service URL: {app.state.embed_url if hasattr(app.state, 'embed_url') else 'Not configured'}")
    logger.info(f"Generation service URL: {app.state.gen_url if hasattr(app.state, 'gen_url') else 'Not configured'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("vLLM OpenAI Service shutting down")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "vLLM OpenAI Service",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "completions": "/v1/completions",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    from src.settings import settings

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.router_port,
        log_level=settings.log_level.lower(),
        reload=False
    )
