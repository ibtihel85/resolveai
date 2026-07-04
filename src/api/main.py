"""
src/api/main.py

FastAPI application entry point for ResolveAI.

Responsibilities:
    - Create the FastAPI app instance
    - Configure middleware (CORS)
    - Register route handlers (chat, health)
    - Manage application lifespan (startup/shutdown)

This file contains NO business logic.
All agent logic lives in src/agent/core.py.
All endpoint logic lives in src/api/routes/*.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.logger import configure_logging, get_logger

# Import routers — we only have chat for now
# Voice, eval routers added in later weeks
from src.api.routes import chat

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Code before yield: runs once at startup.
    Code after yield:  runs once at shutdown.

    Startup order matters:
        1. Logging first — everything after this can log
        2. DB tables — ensure schema exists before requests arrive
        3. Log ready — signal that startup completed successfully
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()

    log.info(
        "resolveai.starting",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        prompt_version=settings.agent_prompt_version,
    )

    # DB table creation — safe to run every startup (creates only if not exists)
    # Imported here to avoid circular imports at module level
    try:
        from src.db.models import create_tables
        create_tables()
        log.info("resolveai.db_ready")
    except Exception as exc:
        # DB failure at startup is logged but does not prevent the app from
        # starting — Week 1 can run without Postgres connected
        log.warning("resolveai.db_unavailable", error=str(exc))

    log.info(
        "resolveai.ready",
        host=settings.api_host,
        port=settings.api_port,
    )

    yield  # ← application is running, handling requests

    # ── Shutdown ──────────────────────────────────────────────────────────────
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
    # Disable docs in production — they expose your API schema publicly
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    # In development: allow all origins
    # In production: replace with your actual frontend domain
    allow_origins=["*"] if not settings.is_production else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

# Health check — lives directly in main.py because it has no business logic
@app.get("/health", tags=["health"])
async def health() -> dict:
    """
    Health check endpoint.
    Used by Docker, load balancers, and monitoring to verify the app is running.
    Returns 200 OK when the application is healthy.
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


# Register the chat router under the /v1/chat prefix
# All chat endpoints defined in src/api/routes/chat.py
app.include_router(
    chat.router,
    prefix="/v1/chat",
    tags=["chat"],
)