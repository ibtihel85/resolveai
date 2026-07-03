"""
src/agent/memory.py

Two-tier conversation memory for ResolveAI:

1. CaseState      — structured facts known about this conversation.
                    Persists for the conversation lifetime.
                    Travels with any human handoff to Zendesk.

2. AgentMemory    — sliding-window buffer of LLM-compatible messages.
                    Keeps only the last N turns to stay within
                    the LLM's context window limit.

These are intentionally separate from src/agent/core.py so that
memory logic can be tested independently with no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config import settings


@dataclass
class CaseState:
    """
    Structured facts about the current conversation.

    Updated as the agent learns more — e.g. once the customer's
    policy is looked up, policy_id is set here so subsequent tool
    calls don't need to ask for it again.

    This object is serialized into the system prompt via
    to_context_string() so the LLM always knows what it already knows.
    """

    conversation_id: str
    channel: str                            # "chat" | "voice"
    customer_id: str | None = None
    policy_id: str | None = None
    customer_name: str | None = None
    open_intents: list[str] = field(default_factory=list)
    resolved_intents: list[str] = field(default_factory=list)
    escalation_flag: bool = False
    escalation_reason: str | None = None
    zendesk_ticket_id: str | None = None
    prompt_version: str = field(
        default_factory=lambda: settings.agent_prompt_version
    )
    started_at: datetime = field(default_factory=datetime.utcnow)

    def to_context_string(self) -> str:
        """
        Compact summary injected into {{ case_context }} in the system prompt.

        The LLM reads this on every turn so it always knows:
        - which conversation this is
        - which channel (chat vs voice affects response style)
        - what customer/policy it is already talking about
        - what intents are still open

        Keeping this compact saves tokens.
        """
        parts = [
            f"conversation_id={self.conversation_id}",
            f"channel={self.channel}",
        ]

        if self.customer_id:
            parts.append(f"customer_id={self.customer_id}")

        if self.customer_name:
            parts.append(f"customer_name={self.customer_name}")

        if self.policy_id:
            parts.append(f"policy_id={self.policy_id}")

        if self.open_intents:
            parts.append(f"open_intents={', '.join(self.open_intents)}")

        if self.escalation_flag:
            parts.append(f"ESCALATED=true reason={self.escalation_reason}")

        return "  |  ".join(parts)


class AgentMemory:
    """
    Sliding-window message buffer for LLM calls.

    Maintains a list of messages in the format the LLM expects:
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "user", "content": "What is my deductible?"},
        ]

    When the buffer exceeds `window` messages, the oldest pair
    (user + assistant) is removed from the front.

    Also holds the CaseState for this conversation.
    """

    def __init__(
        self,
        case_state: CaseState,
        window: int | None = None,
    ) -> None:
        self.case_state = case_state
        self._window = window or settings.agent_memory_window
        self._messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        """Add a user message to the buffer."""
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Add an assistant message to the buffer."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return the current message list for the LLM call.
        Returns a copy so callers cannot accidentally mutate the buffer.
        """
        return list(self._messages)

    def turn_count(self) -> int:
        """Number of complete user turns in the current window."""
        return sum(1 for m in self._messages if m["role"] == "user")

    def _trim(self) -> None:
        """
        Remove the oldest message pair when the buffer exceeds the window.

        Always removes in pairs (user + assistant) to avoid sending
        an assistant message with no preceding user message — which
        would confuse the LLM.
        """
        while len(self._messages) > self._window:
            # Remove the two oldest messages (one pair)
            self._messages = self._messages[2:]