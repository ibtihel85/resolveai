"""
src/logger.py

Structured logging setup for ResolveAI.

Call configure_logging() once at application startup (in api/main.py).
Then call get_logger(__name__) in any module that needs to log.

Two output modes controlled by LOG_FORMAT in .env:
  - "pretty" : coloured human-readable output for local development
  - "json"   : one JSON object per line for production log aggregators
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.config import settings


def configure_logging() -> None:
    """
    Configure structlog for the entire application.
    Call this exactly once — at the top of api/main.py lifespan.
    """

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Get a named logger for a module.

    Usage in any module:
        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("something.happened", key="value", count=42)

    The __name__ argument automatically sets the logger name to the
    module path — e.g. "src.agent.core" or "src.agent.tools.zendesk_tool".
    This makes it easy to filter logs by module in production.
    """
    return structlog.get_logger(name)