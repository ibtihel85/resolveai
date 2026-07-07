"""
src/agent/tools/__init__.py

Tool registry for ResolveAI.

Two responsibilities:
1. TOOL_DEFINITIONS — the list of tool schemas sent to the LLM on every
                      call so it knows what tools exist and when to use them.

2. dispatch()       — called by core.py when the LLM requests a tool call.
                      Maps tool name → async handler function.

To add a new tool:
    1. Create src/agent/tools/my_tool.py with TOOL_DEFINITION and run()
    2. Import it here
    3. Add TOOL_DEFINITION to TOOL_DEFINITIONS
    4. Add "tool_name": my_tool.run to _HANDLERS
    That's it. core.py never needs to change.
"""

from __future__ import annotations

from typing import Any

import src.agent.tools.calendar_tool as calendar_tool
import src.agent.tools.knowledge_base_tool as knowledge_base_tool
import src.agent.tools.policy_crm_tool as policy_crm_tool
import src.agent.tools.slack_tool as slack_tool
import src.agent.tools.zendesk_tool as zendesk_tool
import src.agent.tools.claims_tool as claims_tool
from src.logger import get_logger

log = get_logger(__name__)

# ── Tool definitions ──────────────────────────────────────────────────────────
# Sent to the LLM on every call so it knows what tools exist.
# Order matters slightly — put higher-priority tools first.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    policy_crm_tool.TOOL_DEFINITION,
    claims_tool.TOOL_DEFINITION, 
    knowledge_base_tool.TOOL_DEFINITION,
    zendesk_tool.TOOL_DEFINITION,
    calendar_tool.TOOL_DEFINITION,
    slack_tool.TOOL_DEFINITION,
]

# ── Handler registry ──────────────────────────────────────────────────────────
_HANDLERS: dict[str, Any] = {
    "lookup_policy": policy_crm_tool.run,
    "get_claim_status": claims_tool.run,
    "search_knowledge_base": knowledge_base_tool.run,
    "create_ticket": zendesk_tool.run,
    "book_callback": calendar_tool.run,
    "notify_slack_escalation": slack_tool.run,
}


async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a tool call to the correct handler.

    Called by src/agent/core.py when the LLM returns a tool_use block.

    Args:
        tool_name:  the tool name string returned by the LLM
        tool_input: the arguments the LLM wants to pass to the tool

    Returns:
        A dict with the tool result — always a dict, never raises.
    """
    handler = _HANDLERS.get(tool_name)

    if handler is None:
        log.warning("tool.unknown", tool_name=tool_name)
        return {
            "error": (
                f"Unknown tool: '{tool_name}'. "
                f"Available: {list(_HANDLERS.keys())}"
            )
        }

    log.info(
        "tool.dispatching",
        tool=tool_name,
        input_keys=list(tool_input.keys()),
    )

    try:
        result = await handler(**tool_input)
        log.info("tool.completed", tool=tool_name)
        return result

    except TypeError as exc:
        log.error("tool.bad_arguments", tool=tool_name, error=str(exc))
        return {
            "error": (
                f"Tool '{tool_name}' received unexpected arguments: {exc}"
            )
        }

    except Exception as exc:
        log.error("tool.failed", tool=tool_name, error=str(exc))
        return {"error": f"Tool '{tool_name}' failed: {exc}"}