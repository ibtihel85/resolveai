"""
src/api/routes/chat.py

Chat API routes — multi-turn text conversations with the ResolveAI agent.

Session management:
    Conversations are stored in an in-memory dict keyed by conversation_id.
    This works for a single server instance (sufficient for Week 1).
    Production multi-instance deployment would use Redis for session storage.

Request flow:
    POST /v1/chat/message
        → get or create ConversationManager
        → call manager.handle_turn(message)
        → return ChatResponse
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.core import ConversationManager
from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

# ── Session store ─────────────────────────────────────────────────────────────
# In-memory dict: conversation_id → ConversationManager instance.
#
# Limitation: sessions are lost on server restart and not shared
# across multiple server instances. Sufficient for Week 1 development.
# Production solution: store sessions in Redis with TTL expiry.
_sessions: dict[str, ConversationManager] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Incoming chat message from the user.

    conversation_id:
        Pass None (or omit) to start a new conversation.
        Pass the ID returned from a previous response to continue.

    customer_id:
        Optional — if the customer is authenticated, pass their ID
        so the agent can use it for CRM lookups without asking.
    """
    message: str = Field(
        ...,                        # ... means required — no default
        min_length=1,
        max_length=4000,
        description="The user's message text.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Continue an existing conversation. Omit to start a new one.",
    )
    customer_id: str | None = Field(
        default=None,
        description="Authenticated customer ID for pre-loading context.",
    )


class ChatResponse(BaseModel):
    """
    Agent response returned to the caller.

    All fields are always present — no Optional fields in responses.
    This makes the API contract predictable for frontend developers.
    """
    conversation_id: str = Field(
        description="Use this in the next request to continue the conversation."
    )
    response: str = Field(
        description="The agent's response text."
    )
    is_escalation: bool = Field(
        description="True if this turn triggered escalation to a human agent."
    )
    escalation_reason: str | None = Field(
        description="Why escalation was triggered, if applicable."
    )
    ticket_id: str | None = Field(
        description="Zendesk ticket ID if a ticket was created."
    )
    tool_calls_count: int = Field(
        description="Number of tool calls made during this turn."
    )
    latency_ms: int = Field(
        description="Total time to generate this response in milliseconds."
    )
    prompt_version: str = Field(
        description="Which prompt version handled this turn."
    )


class SessionStateResponse(BaseModel):
    """Current case state for a conversation — used for QA and debugging."""
    conversation_id: str
    customer_id: str | None
    policy_id: str | None
    customer_name: str | None
    open_intents: list[str]
    escalation_flag: bool
    escalation_reason: str | None
    zendesk_ticket_id: str | None
    turn_count: int
    prompt_version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest) -> ChatResponse:
    """
    Send a message to the ResolveAI agent and receive a response.

    Start a new conversation by omitting conversation_id.
    Continue an existing conversation by passing the conversation_id
    returned in the previous response.

    The agent maintains full conversation memory across turns.
    """
    # ── Get or create session ─────────────────────────────────────────────────
    conv_id = req.conversation_id or str(uuid.uuid4())

    if conv_id not in _sessions:
        manager = ConversationManager(
            conversation_id=conv_id,
            channel="chat",
        )
        # Pre-load customer ID if provided by authenticated caller
        if req.customer_id:
            manager.case_state.customer_id = req.customer_id

        _sessions[conv_id] = manager
        log.info("chat.session_created", conversation_id=conv_id)
    else:
        manager = _sessions[conv_id]
        log.info("chat.session_resumed", conversation_id=conv_id)

    # ── Run the agent turn ────────────────────────────────────────────────────
    result = await manager.handle_turn(req.message)

    # ── Clean up escalated conversations ──────────────────────────────────────
    # Once escalated, the agent hands off to a human.
    # The session is no longer needed — free the memory.
    if result.is_escalation:
        _sessions.pop(conv_id, None)
        log.info("chat.session_escalated", conversation_id=conv_id)

    # ── Build and return response ─────────────────────────────────────────────
    return ChatResponse(
        conversation_id=conv_id,
        response=result.response_text,
        is_escalation=result.is_escalation,
        escalation_reason=result.escalation_reason,
        ticket_id=manager.case_state.zendesk_ticket_id,
        tool_calls_count=len(result.tool_calls),
        latency_ms=result.latency_ms,
        prompt_version=settings.agent_prompt_version,
    )


@router.get("/session/{conversation_id}/state", response_model=SessionStateResponse)
async def get_session_state(conversation_id: str) -> SessionStateResponse:
    """
    Return the current case state for a conversation.

    Used by:
        - QA engineers debugging agent behavior
        - The analytics dashboard
        - Integration tests verifying state was updated correctly

    Returns 404 if the conversation does not exist or has ended.
    """
    manager = _sessions.get(conversation_id)

    if not manager:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found or has ended.",
        )

    cs = manager.case_state
    return SessionStateResponse(
        conversation_id=cs.conversation_id,
        customer_id=cs.customer_id,
        policy_id=cs.policy_id,
        customer_name=cs.customer_name,
        open_intents=cs.open_intents,
        escalation_flag=cs.escalation_flag,
        escalation_reason=cs.escalation_reason,
        zendesk_ticket_id=cs.zendesk_ticket_id,
        turn_count=manager.memory.turn_count(),
        prompt_version=cs.prompt_version,
    )


@router.delete("/session/{conversation_id}")
async def end_session(conversation_id: str) -> dict[str, Any]:
    """
    Explicitly end a conversation and free its memory.

    Called when:
        - The user closes the chat window
        - A frontend detects the conversation is complete
        - An automated test cleans up after itself
    """
    existed = conv_id in _sessions if (conv_id := conversation_id) else False
    _sessions.pop(conversation_id, None)

    log.info("chat.session_ended", conversation_id=conversation_id)
    return {
        "status": "ended",
        "conversation_id": conversation_id,
        "existed": existed,
    }