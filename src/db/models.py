"""
src/db/models.py

SQLAlchemy ORM models for conversation logging and analytics.

Two tables:
    conversations       — one row per conversation session
    conversation_turns  — one row per agent response turn

These tables power:
    - src/analytics/logger.py   (writes every turn)
    - src/analytics/dashboard.py (reads for visualisation)
    - evaluation/eval_harness.py (reads for quality analysis)

Database connection is managed via the engine and SessionLocal factory.
Call create_tables() once at application startup (done in api/main.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import settings

# ── Database engine ───────────────────────────────────────────────────────────
# pool_pre_ping=True verifies connections before use — handles dropped
# connections after PostgreSQL restarts without crashing the application.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factory — call SessionLocal() to get a database session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ── Base class ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Conversation(Base):
    """
    One row per conversation session.
    Aggregates metrics across all turns for analytics queries.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    channel: Mapped[str] = mapped_column(String(20))        # "chat" | "voice"
    customer_id: Mapped[str | None] = mapped_column(String(100))
    policy_id: Mapped[str | None] = mapped_column(String(100))
    customer_name: Mapped[str | None] = mapped_column(String(200))
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(100))
    zendesk_ticket_id: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str] = mapped_column(String(20))
    total_turns: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)


class ConversationTurn(Base):
    """
    One row per agent response turn.
    Stores the full detail of every message, tool call, and outcome.
    """

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(String(36))   # FK to conversations
    turn_index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    role: Mapped[str] = mapped_column(String(20))              # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)                 # raw message text
    content_redacted: Mapped[str | None] = mapped_column(Text) # PII-scrubbed version

    # Intent and confidence from guardrails classification
    intent: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)

    # Tool calls made during this turn — stored as JSON list
    # Format: [{"name": "lookup_policy", "input": {...}, "result": {...}, "latency_ms": 340}]
    tool_calls: Mapped[list | None] = mapped_column(JSON)

    # Retrieval results from knowledge base searches
    # Format: [{"doc_id": "kb-001", "score": 0.87, "title": "..."}]
    retrieval_docs: Mapped[list | None] = mapped_column(JSON)

    # Outcome flags
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    is_escalation: Mapped[bool] = mapped_column(Boolean, default=False)
    guardrail_triggered: Mapped[str | None] = mapped_column(String(100))

    # Performance metrics
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class PromptVersion(Base):
    """
    Tracks prompt versions and their evaluation results.
    Updated by the eval harness after each evaluation run.
    """

    __tablename__ = "prompt_versions"

    version: Mapped[str] = mapped_column(String(20), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    description: Mapped[str] = mapped_column(Text)
    task_success_rate: Mapped[float | None] = mapped_column(Float)
    hallucination_rate: Mapped[float | None] = mapped_column(Float)
    avg_quality_score: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


# ── Database utilities ────────────────────────────────────────────────────────

def create_tables() -> None:
    """
    Create all tables if they do not already exist.
    Safe to call on every application startup.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Dependency function for FastAPI route handlers.
    Yields a database session and ensures it is closed after the request.

    Usage in a route:
        from fastapi import Depends
        from src.db.models import get_db

        @router.post("/example")
        async def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()