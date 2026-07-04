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

from src.logger import get_logger

log = get_logger(__name__)

# ── Tool imports ──────────────────────────────────────────────────────────────
# We import lazily inside dispatch() for tools not yet written,
# but for registered tools we import at module level for clarity.
from src.agent.tools import knowledge_base_tool, policy_crm_tool


# ── Tool definitions ──────────────────────────────────────────────────────────
# This list is sent to the LLM on every call.
# The LLM reads name + description to decide which tool to call.
# Order matters slightly — put higher-priority tools first.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    policy_crm_tool.TOOL_DEFINITION,
    knowledge_base_tool.TOOL_DEFINITION,
]


# ── Handler registry ──────────────────────────────────────────────────────────
# Maps tool name (string) → async function that runs the tool.
# core.py calls dispatch(name, input) without knowing which module handles it.
_HANDLERS: dict[str, Any] = {
    "lookup_policy": policy_crm_tool.run,
    "search_knowledge_base": knowledge_base_tool.run,
}


async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a tool call to the correct handler.

    Called by src/agent/core.py when the LLM returns a tool_use block.

    Args:
        tool_name:  the tool name string returned by the LLM
                    e.g. "lookup_policy" or "search_knowledge_base"
        tool_input: the arguments the LLM wants to pass to the tool
                    e.g. {"policy_id": "POL-0023412"}

    Returns:
        A dict with the tool result — always a dict, never raises.
        On unknown tool: returns an error dict so the agent can handle it.

    Design note:
        We never raise exceptions from dispatch(). If a tool fails,
        we return {"error": "..."} so the agent loop can decide whether
        to retry or escalate. Exceptions in tool calls should not crash
        the entire conversation.
    """
    handler = _HANDLERS.get(tool_name)

    if handler is None:
        log.warning("tool.unknown", tool_name=tool_name)
        return {"error": f"Unknown tool: '{tool_name}'. Available: {list(_HANDLERS.keys())}"}

    log.info("tool.dispatching", tool=tool_name, input_keys=list(tool_input.keys()))

    try:
        result = await handler(**tool_input)
        log.info("tool.completed", tool=tool_name)
        return result

    except TypeError as exc:
        # Wrong arguments passed by LLM — common when prompt is unclear
        log.error("tool.bad_arguments", tool=tool_name, error=str(exc))
        return {"error": f"Tool '{tool_name}' received unexpected arguments: {exc}"}

    except Exception as exc:
        # Unexpected failure — log it but don't crash the conversation
        log.error("tool.failed", tool=tool_name, error=str(exc))
        return {"error": f"Tool '{tool_name}' failed: {exc}"}