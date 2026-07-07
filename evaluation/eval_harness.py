"""
evaluation/eval_harness.py

Offline evaluation harness for ResolveAI.

Runs a dataset of golden conversations through the agent and measures:
    - Task success rate (did the conversation reach the expected outcome?)
    - Escalation accuracy (did escalations trigger when expected?)
    - Keyword match rate (did responses contain expected content?)
    - Tool accuracy (did the agent call the expected tool?)
    - Average latency and cost per conversation

Usage:
    python -m evaluation.eval_harness
    python -m evaluation.eval_harness --dataset golden_conversations
    python -m evaluation.eval_harness --prompt-version v1
    python -m evaluation.eval_harness --prompt-version v2

Results are saved to evaluation/reports/ as JSON files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import ConversationManager
from src.config import settings
from src.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TurnEval:
    turn_index: int
    user_message: str
    agent_response: str
    expected_keywords: list[str]
    expected_tool: str | None
    actual_tools_called: list[str]
    keyword_match: bool
    tool_match: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int


@dataclass
class ScenarioEval:
    name: str
    expected_outcome: str
    actual_outcome: str
    outcome_match: bool
    turns: list[TurnEval] = field(default_factory=list)
    total_latency_ms: int = 0
    total_tokens: int = 0
    error: str | None = None


@dataclass
class EvalReport:
    dataset: str
    prompt_version: str
    run_at: str
    total_scenarios: int
    task_success_rate: float
    escalation_accuracy: float
    keyword_match_rate: float
    tool_accuracy_rate: float
    avg_latency_ms: float
    total_tokens: int
    scenarios: list[ScenarioEval] = field(default_factory=list)


# ── Outcome detection ─────────────────────────────────────────────────────────

def _detect_outcome(manager: ConversationManager, last_result) -> str:
    """
    Determine the actual outcome of a conversation.
    Returns "escalated", "blocked", or "resolved".
    """
    if manager.case_state.escalation_flag:
        return "escalated"
    if last_result and last_result.is_fallback:
        return "blocked"
    return "resolved"


# ── Single scenario runner ────────────────────────────────────────────────────

async def run_scenario(
    scenario: dict,
    prompt_version: str,
) -> ScenarioEval:
    """Run one golden conversation scenario and evaluate the results."""

    name = scenario.get("name", "unnamed")
    expected_outcome = scenario.get("expected_outcome", "resolved")
    turns_data = scenario.get("turns", [])

    log.info("eval.scenario_started", name=name)

    # Override prompt version for this run
    original_version = settings.agent_prompt_version
    settings.agent_prompt_version = prompt_version

    manager = ConversationManager(channel="chat")
    turn_evals: list[TurnEval] = []
    total_latency = 0
    total_tokens = 0
    last_result = None
    error = None

    try:
        for i, turn_data in enumerate(turns_data):
            user_message = turn_data.get("user", "")
            expected_keywords = turn_data.get("expected_keywords", [])
            expected_tool = turn_data.get("expected_tool")

            result = await manager.handle_turn(user_message)
            last_result = result

            # Check keyword match
            response_lower = result.response_text.lower()
            keyword_match = all(
                kw.lower() in response_lower
                for kw in expected_keywords
            ) if expected_keywords else True

            # Check tool match
            actual_tools = [tc["name"] for tc in result.tool_calls]
            tool_match = (
                expected_tool in actual_tools
                if expected_tool
                else True
            )

            turn_evals.append(TurnEval(
                turn_index=i,
                user_message=user_message,
                agent_response=result.response_text,
                expected_keywords=expected_keywords,
                expected_tool=expected_tool,
                actual_tools_called=actual_tools,
                keyword_match=keyword_match,
                tool_match=tool_match,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ))

            total_latency += result.latency_ms
            total_tokens += result.input_tokens + result.output_tokens

    except Exception as exc:
        error = str(exc)
        log.error("eval.scenario_error", name=name, error=error)

    finally:
        settings.agent_prompt_version = original_version

    actual_outcome = _detect_outcome(manager, last_result)
    outcome_match = actual_outcome == expected_outcome

    status = "✓" if outcome_match else "✗"
    print(f"  {status} {name:<45} expected={expected_outcome:<10} actual={actual_outcome}")

    return ScenarioEval(
        name=name,
        expected_outcome=expected_outcome,
        actual_outcome=actual_outcome,
        outcome_match=outcome_match,
        turns=turn_evals,
        total_latency_ms=total_latency,
        total_tokens=total_tokens,
        error=error,
    )


# ── Main evaluation runner ────────────────────────────────────────────────────

async def run_evaluation(
    dataset_name: str = "golden_conversations",
    prompt_version: str | None = None,
) -> EvalReport:
    """
    Run all scenarios in a dataset and produce an EvalReport.
    """
    prompt_version = prompt_version or settings.agent_prompt_version

    dataset_path = DATASETS_DIR / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    scenarios = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    print(f"\n{'='*65}")
    print(f"  ResolveAI Evaluation Harness")
    print(f"  Dataset:        {dataset_name}")
    print(f"  Prompt version: {prompt_version}")
    print(f"  Scenarios:      {len(scenarios)}")
    print(f"{'='*65}")

    results: list[ScenarioEval] = []
    for scenario in scenarios:
        result = await run_scenario(scenario, prompt_version)
        results.append(result)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    success_count = sum(1 for r in results if r.outcome_match)
    task_success_rate = success_count / total if total else 0.0

    # Escalation accuracy — of scenarios expecting escalation, how many escalated?
    expected_escalations = [r for r in results if r.expected_outcome == "escalated"]
    correct_escalations = [r for r in expected_escalations if r.outcome_match]
    escalation_accuracy = (
        len(correct_escalations) / len(expected_escalations)
        if expected_escalations else 1.0
    )

    # Keyword match rate across all turns
    all_turns = [t for r in results for t in r.turns]
    keyword_turns = [t for t in all_turns if t.expected_keywords]
    keyword_match_rate = (
        sum(1 for t in keyword_turns if t.keyword_match) / len(keyword_turns)
        if keyword_turns else 1.0
    )

    # Tool accuracy across all turns with expected tools
    tool_turns = [t for t in all_turns if t.expected_tool]
    tool_accuracy_rate = (
        sum(1 for t in tool_turns if t.tool_match) / len(tool_turns)
        if tool_turns else 1.0
    )

    avg_latency = (
        sum(r.total_latency_ms for r in results) / total
        if total else 0.0
    )
    total_tokens = sum(r.total_tokens for r in results)

    report = EvalReport(
        dataset=dataset_name,
        prompt_version=prompt_version,
        run_at=datetime.utcnow().isoformat(),
        total_scenarios=total,
        task_success_rate=round(task_success_rate, 4),
        escalation_accuracy=round(escalation_accuracy, 4),
        keyword_match_rate=round(keyword_match_rate, 4),
        tool_accuracy_rate=round(tool_accuracy_rate, 4),
        avg_latency_ms=round(avg_latency),
        total_tokens=total_tokens,
        scenarios=results,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Results — prompt {prompt_version}")
    print(f"{'='*65}")
    print(f"  Task success rate:    {task_success_rate:.1%}  ({success_count}/{total})")
    print(f"  Escalation accuracy:  {escalation_accuracy:.1%}")
    print(f"  Keyword match rate:   {keyword_match_rate:.1%}")
    print(f"  Tool accuracy:        {tool_accuracy_rate:.1%}")
    print(f"  Avg latency:          {avg_latency:.0f} ms")
    print(f"  Total tokens:         {total_tokens:,}")
    print(f"{'='*65}")

    # ── Save report ───────────────────────────────────────────────────────────
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_{dataset_name}_{prompt_version}_{ts}.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, default=str)
    )
    print(f"  Report saved: {report_path}\n")

    return report


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResolveAI evaluation harness")
    parser.add_argument(
        "--dataset",
        default="golden_conversations",
        help="Dataset name (without .jsonl extension)",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt version to evaluate (default: from .env)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            dataset_name=args.dataset,
            prompt_version=args.prompt_version,
        )
    )