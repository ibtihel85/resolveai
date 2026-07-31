"""
src/api/main.py

FastAPI application entry point for ResolveAI.

Responsibilities:
    - Create the FastAPI app instance
    - Configure middleware (CORS)
    - Register route handlers (chat, voice, health)
    - Manage application lifespan (startup/shutdown)

This file contains NO business logic.
All agent logic lives in src/agent/core.py.
All endpoint logic lives in src/api/routes/*.py.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, voice
from src.config import settings
from src.logger import configure_logging, get_logger

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Code before yield: runs once at startup.
    Code after yield:  runs once at shutdown.

    Startup order:
        1. Logging first — everything after this can log
        2. DB tables — ensure schema exists before requests arrive
        3. Log ready — signal that startup completed successfully
    """
    configure_logging()

    log.info(
        "resolveai.starting",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        prompt_version=settings.agent_prompt_version,
    )

    try:
        from src.db.models import create_tables
        create_tables()
        log.info("resolveai.db_ready")
    except Exception as exc:
        log.warning("resolveai.db_unavailable", error=str(exc))

    log.info(
        "resolveai.ready",
        host=settings.api_host,
        port=settings.api_port,
    )

    yield

    log.info("resolveai.shutting_down")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResolveAI — Insurance Support Agent API",
    version="0.1.0",
    description=(
        "Multi-channel enterprise AI support agent for Meridian Insurance. "
        "Handles policy queries, claims status, ticketing, and voice interactions."
    ),
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health() -> dict:
    """
    Health check endpoint.
    Used by Docker, load balancers, and monitoring to verify the app is running.
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


app.include_router(
    chat.router,
    prefix="/v1/chat",
    tags=["chat"],
)

app.include_router(
    voice.router,
    prefix="/v1/voice",
    tags=["voice"],
)