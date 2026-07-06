"""
src/analytics/logger.py

Conversation analytics logger — writes every agent turn to PostgreSQL.

Called by src/api/routes/chat.py after each handle_turn() call.
Data written here powers the Streamlit analytics dashboard.

Two write operations per turn:
    1. Upsert the Conversation row (create or update aggregated metrics)
    2. Insert a ConversationTurn row (full detail of this specific turn)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.agent.core import TurnResult
from src.agent.memory import CaseState
from src.config import settings
from src.db.models import Conversation, ConversationTurn
from src.logger import get_logger

log = get_logger(__name__)

# ── Token cost constants ──────────────────────────────────────────────────────
# Approximate costs per 1,000 tokens.
# For local Ollama models, cost is 0 — constants kept for provider portability.
# Update these when switching to a paid provider.
_INPUT_COST_PER_1K = 0.00025    # USD per 1K input tokens (Claude Haiku reference)
_OUTPUT_COST_PER_1K = 0.00125   # USD per 1K output tokens (Claude Haiku reference)


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate approximate USD cost for a turn based on token usage."""
    if settings.llm_provider == "ollama":
        return 0.0
    return (
        input_tokens / 1000 * _INPUT_COST_PER_1K
        + output_tokens / 1000 * _OUTPUT_COST_PER_1K
    )


class ConversationLogger:
    """
    Writes conversation turns to PostgreSQL for analytics.

    One instance per request — instantiated in the chat route handler
    with an injected database session.

    Usage:
        logger = ConversationLogger(db)
        await logger.log_turn(conversation_id, channel, case_state, message, result)
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    async def log_turn(
        self,
        conversation_id: str,
        channel: str,
        case_state: CaseState,
        user_message: str,
        result: TurnResult,
        turn_index: int,
    ) -> None:
        """
        Write a single agent turn to the database.

        Creates the Conversation row on the first turn.
        Updates aggregated metrics on subsequent turns.
        Always inserts a new ConversationTurn row.

        Errors are caught and logged — a logging failure must never
        crash the conversation or return an error to the user.
        """
        try:
            cost = _calculate_cost(result.input_tokens, result.output_tokens)

            # ── Upsert conversation row ───────────────────────────────────────
            conv = self._db.get(Conversation, conversation_id)

            if conv is None:
                conv = Conversation(
                    id=conversation_id,
                    channel=channel,
                    customer_id=case_state.customer_id,
                    policy_id=case_state.policy_id,
                    customer_name=case_state.customer_name,
                    prompt_version=settings.agent_prompt_version,
                )
                self._db.add(conv)
                log.info("analytics.conversation_created", conversation_id=conversation_id)

            # Update aggregated metrics
            conv.total_turns = turn_index
            conv.total_tokens += result.input_tokens + result.output_tokens
            conv.total_cost_usd += cost
            conv.total_latency_ms += result.latency_ms

            # Update case state fields that may have been resolved this turn
            if case_state.customer_id:
                conv.customer_id = case_state.customer_id
            if case_state.policy_id:
                conv.policy_id = case_state.policy_id
            if case_state.customer_name:
                conv.customer_name = case_state.customer_name

            # Mark escalation on the conversation row
            if result.is_escalation:
                conv.escalated = True
                conv.escalation_reason = result.escalation_reason
                conv.ended_at = datetime.utcnow()
                if case_state.zendesk_ticket_id:
                    conv.zendesk_ticket_id = case_state.zendesk_ticket_id

            # ── Insert turn row ───────────────────────────────────────────────
            turn = ConversationTurn(
                conversation_id=conversation_id,
                turn_index=turn_index,
                role="assistant",
                content=result.response_text,
                tool_calls=result.tool_calls if result.tool_calls else None,
                is_fallback=result.is_fallback,
                is_escalation=result.is_escalation,
                guardrail_triggered=result.guardrail_triggered
                    if hasattr(result, "guardrail_triggered") else None,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=cost,
            )
            self._db.add(turn)
            self._db.commit()

            log.info(
                "analytics.turn_logged",
                conversation_id=conversation_id,
                turn_index=turn_index,
                tokens=result.input_tokens + result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=cost,
            )

        except Exception as exc:
            # Never let logging failures affect the conversation
            self._db.rollback()
            log.error(
                "analytics.log_failed",
                conversation_id=conversation_id,
                error=str(exc),
            )