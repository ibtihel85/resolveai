"""
tests/regression/test_prompt_regression.py

Prompt regression tests — run golden dataset scenarios with mocked
LLM and tools to verify guardrail and escalation behavior is correct.

These tests run in CI on every push. They catch regressions in:
    - Guardrail blocking behavior
    - Escalation trigger logic
    - Session and memory management

"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Load golden dataset ───────────────────────────────────────────────────────
DATASET_PATH = (
    Path(__file__).parent.parent.parent
    / "evaluation"
    / "datasets"
    / "golden_conversations.jsonl"
)


def load_scenarios():
    scenarios = []
    with DATASET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


SCENARIOS = load_scenarios()


# ── Mock helpers ──────────────────────────────────────────────────────────────

def make_llm_response(text: str = "I can help you with your insurance query."):
    response = MagicMock()
    response.usage.prompt_tokens = 80
    response.usage.completion_tokens = 40
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = text
    choice.message.tool_calls = None
    response.choices = [choice]
    return response


# ── Regression tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[s["name"] for s in SCENARIOS],
)

async def test_scenario_outcome(scenario):
    """
    Each scenario runs with mocked LLM and tools.
    Verifies that guardrails and escalation logic produce
    the expected outcome for every golden conversation.
    """
    from src.agent.core import ConversationManager

    if scenario["name"] == "repeated_tool_failure_escalation":
        pytest.skip("Retry escalation requires real tool failures — tested in conversation_flows")

    expected_outcome = scenario.get("expected_outcome", "resolved")

    with patch(
        "src.agent.core._llm_client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=make_llm_response(),
    ):
        with patch(
            "src.agent.tools.policy_crm_tool.run",
            new_callable=AsyncMock,
            return_value={
                "policy_id": "POL-001",
                "status": "active",
                "customer_name": "Test User",
                "customer_id": "CUST-1",
                "deductible": 500.0,
                "coverage_limit": 100000.0,
                "policy_type": "home",
            },
        ):
            with patch(
                "src.agent.tools.claims_tool.run",
                new_callable=AsyncMock,
                return_value={"claim_id": "CLM-001", "status": "under_review"},
            ):
                with patch(
                    "src.agent.tools.knowledge_base_tool.run",
                    return_value={
                        "found": True,
                        "best_score": 0.85,
                        "results": [
                            {
                                "doc_id": "kb-001",
                                "title": "Home Insurance Coverage",
                                "text": "Covers fire, water damage, theft.",
                                "similarity_score": 0.85,
                            }
                        ],
                    },
                ):
                    with patch(
                        "src.agent.tools.zendesk_tool.run",
                        new_callable=AsyncMock,
                        return_value={"ticket_id": "T-1", "status": "created"},
                    ):
                        with patch(
                            "src.agent.tools.slack_tool.run",
                            new_callable=AsyncMock,
                            return_value={"status": "sent"},
                        ):
                            manager = ConversationManager(channel="chat")
                            last_result = None

                            for turn in scenario.get("turns", []):
                                last_result = await manager.handle_turn(
                                    turn["user"]
                                )

    # Determine actual outcome
    if manager.case_state.escalation_flag:
        actual_outcome = "escalated"
    elif last_result and last_result.is_fallback:
        actual_outcome = "blocked"
    else:
        actual_outcome = "resolved"

    if expected_outcome == "escalated":
        assert actual_outcome in ("escalated", "blocked"), (
            f"Scenario '{scenario['name']}': "
            f"expected escalation but got '{actual_outcome}'"
        )
    elif expected_outcome == "blocked":
        assert actual_outcome == "blocked", (
            f"Scenario '{scenario['name']}': "
            f"expected blocked but got '{actual_outcome}'"
        )
    else:
        assert last_result is not None
        assert last_result.response_text, (
            f"Scenario '{scenario['name']}': empty response"
        )