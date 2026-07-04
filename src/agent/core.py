"""
src/agent/core.py

Agent orchestration core — the conversation manager and ReAct loop.

This file connects every other component:
    - prompts/   → system prompt loaded and rendered here
    - memory.py  → sliding window and case state managed here
    - tools/     → tool definitions sent to LLM, dispatch called here
    - logger.py  → every turn logged with full metrics
    - config.py  → LLM provider, model, temperature all read here

The ReAct loop (Reasoning + Acting):
    1. Send system prompt + message history + tool definitions to LLM
    2. LLM either responds with text (done) or requests a tool call
    3. If tool call: dispatch it, add result to messages, loop back
    4. If text: return the response
    5. After max_rounds: escalate rather than loop forever
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import openai

from src.agent.memory import AgentMemory, CaseState
from src.agent.prompts import get_system_prompt
from src.agent.tools import TOOL_DEFINITIONS, dispatch
from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)


# ── LLM client ────────────────────────────────────────────────────────────────
# Built once at module load time — reused across all requests.
# The same openai.AsyncOpenAI client works for both Ollama and Anthropic
# because Ollama exposes an OpenAI-compatible API.

def _build_llm_client() -> openai.AsyncOpenAI:
    """Build the LLM client based on the configured provider."""
    if settings.llm_provider == "ollama":
        return openai.AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",          # Ollama ignores the key but requires one
        )
    elif settings.llm_provider == "anthropic":
        return openai.AsyncOpenAI(
            api_key=settings.anthropic_api_key,
            base_url="https://api.anthropic.com/v1",
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{settings.llm_provider}'. "
            "Set LLM_PROVIDER=ollama or LLM_PROVIDER=anthropic in .env"
        )


_llm_client = _build_llm_client()


# ── Turn result ───────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """
    Everything produced by a single agent turn.
    Returned to the API layer and used for logging and analytics.
    """
    response_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    is_fallback: bool = False
    is_escalation: bool = False
    escalation_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


# ── Canned responses ──────────────────────────────────────────────────────────
# Used when guardrails block a request before the LLM is called.
# Defined here so they're easy to find and update.

_BLOCK_RESPONSES = {
    "prompt_injection": (
        "I'm not able to process that request. "
        "Is there something I can help you with regarding your Meridian Insurance policy?"
    ),
    "out_of_scope": (
        "That's outside what I can help with here. "
        "I specialise in policy questions, claims, billing, and appointments. "
        "Can I help you with any of those?"
    ),
}

_ESCALATION_RESPONSES = {
    "customer_requested_human": (
        "Of course — let me connect you with a team member right away. "
        "I'll make sure they have full context on your situation."
    ),
    "anger_detected": (
        "I'm sorry you're having such a frustrating experience. "
        "Let me get you connected with a team member immediately."
    ),
    "repeated_tool_failure": (
        "I'm running into a technical issue on my end. "
        "Let me escalate this to a team member who can assist you directly."
    ),
    "max_rounds_exceeded": (
        "I want to make sure you get the right help here. "
        "Let me connect you with a team member who can take this further."
    ),
}


# ── Conversation manager ──────────────────────────────────────────────────────

class ConversationManager:
    """
    Manages a single conversation end-to-end.

    One instance per conversation — created when a new conversation
    starts, held in memory for its duration, discarded when it ends.

    Responsibilities:
        - Hold the AgentMemory (sliding window + case state)
        - Run the ReAct loop on each user turn
        - Track retry counts for escalation decisions
        - Log every turn with full metrics
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        channel: str = "chat",
    ) -> None:
        self.conversation_id = conversation_id or str(uuid.uuid4())
        case_state = CaseState(
            conversation_id=self.conversation_id,
            channel=channel,
        )
        self.memory = AgentMemory(case_state)
        self._retry_count: int = 0

    @property
    def case_state(self) -> CaseState:
        """Convenience accessor — case state lives inside memory."""
        return self.memory.case_state

    async def handle_turn(self, user_message: str) -> TurnResult:
        """
        Process one user turn and return a TurnResult.

        This is the main entry point called by the API layer (chat.py).
        It runs the full ReAct loop and returns whatever the agent produces.

        Args:
            user_message: the raw text from the user

        Returns:
            TurnResult with response text, tool calls made, and metrics.
        """
        t0 = time.monotonic()

        log.info(
            "agent.turn_started",
            conversation_id=self.conversation_id,
            turn=self.memory.turn_count() + 1,
            message_preview=user_message[:60],
        )

        # Add user message to memory before the LLM call
        self.memory.add_user(user_message)

        # Run the ReAct loop
        result = await self._react_loop(t0)

        # Add assistant response to memory for next turn's context
        self.memory.add_assistant(result.response_text)

        log.info(
            "agent.turn_completed",
            conversation_id=self.conversation_id,
            latency_ms=result.latency_ms,
            tool_calls=len(result.tool_calls),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            is_escalation=result.is_escalation,
        )

        return result

    async def _react_loop(self, t0: float) -> TurnResult:
        """
        The ReAct loop — Reasoning + Acting.

        Calls the LLM, dispatches tool calls if requested,
        feeds results back, repeats until the LLM returns text
        or we hit the maximum number of rounds.

        Args:
            t0: start time from time.monotonic() for latency tracking

        Returns:
            TurnResult with the final response and all metrics.
        """
        tool_calls_log: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        max_rounds = 5

        # Build current message history for the LLM
        messages = self.memory.get_messages()

        # Build system prompt with current case context injected
        system_prompt = get_system_prompt(
            version=settings.agent_prompt_version,
            case_context=self.case_state.to_context_string(),
        )

        for round_num in range(max_rounds):
            log.info(
                "agent.llm_call",
                conversation_id=self.conversation_id,
                round=round_num + 1,
                messages_in_context=len(messages),
            )

            # ── Call the LLM ──────────────────────────────────────────────────
            response = await _llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",        # LLM decides whether to use a tool
                max_tokens=settings.agent_max_tokens,
                temperature=settings.agent_temperature,
            )

            # Track token usage for cost/analytics logging
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens

            choice = response.choices[0]

            # ── Case A: LLM returned text — we are done ───────────────────────
            if choice.finish_reason == "stop":
                response_text = choice.message.content or ""
                self._retry_count = 0   # reset on successful turn

                return TurnResult(
                    response_text=response_text,
                    tool_calls=tool_calls_log,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            # ── Case B: LLM wants to call tools ──────────────────────────────
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Add the assistant's tool-call message to the conversation
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                ]

                # Dispatch each tool call and collect results
                import json
                tool_result_messages = []

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}

                    log.info(
                        "agent.tool_call",
                        tool=tool_name,
                        conversation_id=self.conversation_id,
                    )

                    tc_start = time.monotonic()
                    tool_result = await dispatch(tool_name, tool_input)
                    tc_ms = int((time.monotonic() - tc_start) * 1000)

                    # Update case state from policy lookups
                    if tool_name == "lookup_policy" and "policy_id" in tool_result:
                        self.case_state.policy_id = tool_result.get("policy_id")
                        self.case_state.customer_id = tool_result.get("customer_id")
                        self.case_state.customer_name = tool_result.get("customer_name")

                    tool_calls_log.append({
                        "name": tool_name,
                        "input": tool_input,
                        "result": tool_result,
                        "latency_ms": tc_ms,
                    })

                    tool_result_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })

                # Add tool results to messages — LLM reads these next round
                messages = messages + tool_result_messages
                continue   # go back to top of loop — call LLM again

            # ── Case C: unexpected finish reason ─────────────────────────────
            log.warning(
                "agent.unexpected_finish",
                finish_reason=choice.finish_reason,
                conversation_id=self.conversation_id,
            )
            break

        # ── Exceeded max rounds — escalate ────────────────────────────────────
        log.warning(
            "agent.max_rounds_exceeded",
            conversation_id=self.conversation_id,
            rounds=max_rounds,
        )
        self._retry_count += 1
        self.case_state.escalation_flag = True
        self.case_state.escalation_reason = "max_rounds_exceeded"

        return TurnResult(
            response_text=_ESCALATION_RESPONSES["max_rounds_exceeded"],
            tool_calls=tool_calls_log,
            is_escalation=True,
            escalation_reason="max_rounds_exceeded",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )