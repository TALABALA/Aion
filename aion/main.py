"""
AION - Artificial Intelligence Operating Nexus

Main entry point for the AION cognitive architecture.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aion.core.config import AIONConfig, get_config, set_config
from aion.core.kernel import AIONKernel
from aion.api.routes import (
    setup_routes,
    setup_planning_routes,
    setup_memory_routes,
    setup_tool_routes,
    setup_evolution_routes,
    setup_vision_routes,
)


# Configure structured logging
def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger(__name__)


# Global kernel instance
kernel: Optional[AIONKernel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global kernel

    logger.info("Starting AION system")

    # Initialize kernel
    config = get_config()
    kernel = AIONKernel(config)
    await kernel.initialize()

    # Setup routes with initialized subsystems
    setup_routes(app, kernel)

    if kernel.planning:
        setup_planning_routes(app, kernel.planning)

    if kernel.memory:
        setup_memory_routes(app, kernel.memory)

    if kernel.tools:
        setup_tool_routes(app, kernel.tools)

    if kernel.evolution:
        setup_evolution_routes(app, kernel.evolution)

    if kernel.vision:
        setup_vision_routes(app, kernel.vision)

    logger.info("AION system ready", status=kernel.get_status())

    yield

    # Shutdown
    logger.info("Shutting down AION system")
    await kernel.shutdown()
    logger.info("AION system shutdown complete")


def create_app(config: Optional[AIONConfig] = None) -> FastAPI:
    """
    Create and configure the AION FastAPI application.

    Args:
        config: Optional configuration override

    Returns:
        Configured FastAPI application
    """
    if config:
        set_config(config)
    else:
        config = get_config()

    # Setup logging
    setup_logging(config.monitoring.log_level.value)

    app = FastAPI(
        title="AION - Artificial Intelligence Operating Nexus",
        description="""
        A production-ready AGI cognitive architecture with:
        - Deterministic Planning Graph
        - Vector Memory Search (FAISS)
        - Tool Orchestration
        - Self-Improvement Loop
        - Visual Cortex

        This API provides access to all AION subsystems for building
        intelligent applications.
        """,
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """
    Run the AION server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    config = get_config()
    config.host = host
    config.port = port
    set_config(config)

    uvicorn.run(
        "aion.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=config.monitoring.log_level.value.lower(),
    )


# Create the default app instance
app = create_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AION - AI Operating Nexus")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
