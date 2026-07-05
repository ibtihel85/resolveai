"""
tests/unit/test_memory.py

Unit tests for src/agent/memory.py

Tests cover:
    - CaseState context string serialization
    - AgentMemory sliding window trimming
    - Message ordering and pair integrity
    - Turn counting
    - Case state updates

No external dependencies — no LLM, no database, no HTTP.
"""

from __future__ import annotations

import pytest

from src.agent.memory import AgentMemory, CaseState


# ── Fixtures ──────────────────────────────────────────────────────────────────
# Fixtures are reusable setup functions shared across tests.
# pytest injects them automatically by parameter name.

@pytest.fixture
def case_state() -> CaseState:
    """Minimal CaseState for testing."""
    return CaseState(
        conversation_id="test-conv-001",
        channel="chat",
    )


@pytest.fixture
def memory(case_state: CaseState) -> AgentMemory:
    """AgentMemory with a small window for easy testing."""
    return AgentMemory(case_state, window=6)


# ── CaseState tests ───────────────────────────────────────────────────────────

class TestCaseState:
    def test_minimal_context_string(self, case_state: CaseState) -> None:
        """Context string always includes conversation_id and channel."""
        ctx = case_state.to_context_string()
        assert "test-conv-001" in ctx
        assert "chat" in ctx

    def test_context_string_includes_customer_id_when_set(
        self, case_state: CaseState
    ) -> None:
        """customer_id appears in context only after it is set."""
        assert "CUST-001" not in case_state.to_context_string()
        case_state.customer_id = "CUST-001"
        assert "CUST-001" in case_state.to_context_string()

    def test_context_string_includes_policy_id_when_set(
        self, case_state: CaseState
    ) -> None:
        """policy_id appears in context only after it is set."""
        assert "POL-0023412" not in case_state.to_context_string()
        case_state.policy_id = "POL-0023412"
        assert "POL-0023412" in case_state.to_context_string()

    def test_context_string_includes_escalation_flag(
        self, case_state: CaseState
    ) -> None:
        """Escalation flag and reason appear in context when set."""
        case_state.escalation_flag = True
        case_state.escalation_reason = "anger_detected"
        ctx = case_state.to_context_string()
        assert "ESCALATED=true" in ctx
        assert "anger_detected" in ctx

    def test_context_string_excludes_none_fields(
        self, case_state: CaseState
    ) -> None:
        """Fields that are None should not appear in the context string."""
        ctx = case_state.to_context_string()
        assert "None" not in ctx
        assert "customer_id" not in ctx
        assert "policy_id" not in ctx

    def test_default_prompt_version(self, case_state: CaseState) -> None:
        """Prompt version defaults to the value in settings."""
        from src.config import settings
        assert case_state.prompt_version == settings.agent_prompt_version


# ── AgentMemory tests ─────────────────────────────────────────────────────────

class TestAgentMemory:
    def test_empty_on_creation(self, memory: AgentMemory) -> None:
        """A new memory has no messages."""
        assert memory.get_messages() == []
        assert memory.turn_count() == 0

    def test_add_user_message(self, memory: AgentMemory) -> None:
        """Adding a user message stores it with correct role."""
        memory.add_user("Hello, I need help.")
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, I need help."

    def test_add_assistant_message(self, memory: AgentMemory) -> None:
        """Adding an assistant message stores it with correct role."""
        memory.add_user("Hello")
        memory.add_assistant("Hi, how can I help?")
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi, how can I help?"

    def test_turn_count_increments_on_user_messages(
        self, memory: AgentMemory
    ) -> None:
        """Turn count reflects the number of user messages only."""
        assert memory.turn_count() == 0
        memory.add_user("message 1")
        assert memory.turn_count() == 1
        memory.add_assistant("response 1")
        assert memory.turn_count() == 1   # assistant doesn't increment count
        memory.add_user("message 2")
        assert memory.turn_count() == 2

    def test_get_messages_returns_copy(self, memory: AgentMemory) -> None:
        """get_messages() returns a copy — mutating it does not affect memory."""
        memory.add_user("hello")
        messages = memory.get_messages()
        messages.clear()
        assert len(memory.get_messages()) == 1   # original unaffected

    def test_sliding_window_trims_oldest_pair(self) -> None:
        """When window is exceeded, the oldest user+assistant pair is removed."""
        cs = CaseState(conversation_id="trim-test", channel="chat")
        memory = AgentMemory(cs, window=4)   # holds 4 messages = 2 pairs

        memory.add_user("turn 1 user")
        memory.add_assistant("turn 1 assistant")
        memory.add_user("turn 2 user")
        memory.add_assistant("turn 2 assistant")
        # Window is now full (4 messages)

        memory.add_user("turn 3 user")
        memory.add_assistant("turn 3 assistant")
        # Window exceeded — turn 1 pair should be dropped

        messages = memory.get_messages()
        assert len(messages) == 4
        # First message should now be turn 2, not turn 1
        assert messages[0]["content"] == "turn 2 user"
        assert messages[1]["content"] == "turn 2 assistant"

    def test_sliding_window_preserves_most_recent(self) -> None:
        """After trimming, the most recent messages are retained."""
        cs = CaseState(conversation_id="recent-test", channel="chat")
        memory = AgentMemory(cs, window=4)

        for i in range(5):
            memory.add_user(f"user {i}")
            memory.add_assistant(f"assistant {i}")

        messages = memory.get_messages()
        # Should have the last 2 pairs (4 messages)
        assert messages[-2]["content"] == "user 4"
        assert messages[-1]["content"] == "assistant 4"

    def test_no_orphaned_assistant_messages(self) -> None:
        """After trimming, the first message is always a user message."""
        cs = CaseState(conversation_id="orphan-test", channel="chat")
        memory = AgentMemory(cs, window=4)

        for i in range(10):
            memory.add_user(f"user {i}")
            memory.add_assistant(f"assistant {i}")

        messages = memory.get_messages()
        assert messages[0]["role"] == "user"

    def test_message_order_preserved(self, memory: AgentMemory) -> None:
        """Messages are returned in the order they were added."""
        memory.add_user("first")
        memory.add_assistant("second")
        memory.add_user("third")
        memory.add_assistant("fourth")

        messages = memory.get_messages()
        assert messages[0]["content"] == "first"
        assert messages[1]["content"] == "second"
        assert messages[2]["content"] == "third"
        assert messages[3]["content"] == "fourth"

    def test_case_state_accessible_through_memory(
        self, memory: AgentMemory, case_state: CaseState
    ) -> None:
        """Memory holds a reference to the case state."""
        assert memory.case_state is case_state
        memory.case_state.policy_id = "POL-001"
        assert case_state.policy_id == "POL-001"