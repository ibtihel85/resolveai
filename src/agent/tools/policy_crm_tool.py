"""
src/agent/tools/policy_crm_tool.py

Policy CRM tool — looks up customer policy data from the mock CRM service.

This tool handles DETERMINISTIC data — there is exactly one correct answer
for any given policy ID. We never use RAG or the knowledge base for this.

Why a separate mock CRM service instead of a database query?
    In production, this would call Salesforce, Guidewire, or a proprietary
    insurer CRM via REST API. The mock service has the same interface shape —
    same auth pattern, same JSON schema, same error codes — so the integration
    code is production-representative even though the data is fake.
    Swapping mock → real CRM is a config change (MOCK_CRM_URL), not a code change.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Tool definition ───────────────────────────────────────────────────────────
TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_policy",
        "description": (
            "Look up a customer's insurance policy from the Meridian Insurance system. "
            "Use this when the customer mentions their policy, coverage details, "
            "deductible, premium, or account information. "
            "Returns live policy data including coverage type, limits, deductible, "
            "premium, and policy status. "
            "Always call this before get_claim_status if you don't have a policy_id yet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": (
                        "Policy identifier, format POL-XXXXXXX. "
                        "Use this if the customer provides it directly."
                    ),
                },
                "customer_id": {
                    "type": "string",
                    "description": (
                        "Customer ID if policy ID is unknown. "
                        "Use this to find all policies for a customer."
                    ),
                },
                "customer_name": {
                    "type": "string",
                    "description": (
                        "Customer full name — use as last resort if neither "
                        "policy_id nor customer_id is known."
                    ),
                },
            },
            # No required fields — LLM provides whatever it knows
        },
    },
}


# ── Tool handler ──────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    reraise=True,    # after all retries fail, re-raise the last exception
)
async def _fetch_policy(params: dict[str, str]) -> dict[str, Any]:
    """
    Internal function — makes the actual HTTP call to the mock CRM.
    Wrapped in retry logic separately from run() so retries only
    cover the network call, not the entire tool logic.
    """
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(
            f"{settings.mock_crm_url}/v1/policies",
            params=params,
            headers={"X-API-Key": settings.mock_crm_api_key},
        )
        response.raise_for_status()
        return response.json()


async def run(
    policy_id: str | None = None,
    customer_id: str | None = None,
    customer_name: str | None = None,
) -> dict[str, Any]:
    """
    Look up a policy from the mock CRM service.

    Called by src/agent/tools/__init__.py dispatch() when the LLM
    requests the lookup_policy tool.

    Args:
        policy_id:     preferred — direct policy lookup
        customer_id:   fallback — find policies for this customer
        customer_name: last resort — fuzzy name search

    Returns:
        dict with policy data on success, or {"error": "..."} on failure.
        Never raises — errors are returned as dicts for the agent to handle.
    """
    # Validate — at least one identifier must be provided
    if not any([policy_id, customer_id, customer_name]):
        return {
            "error": (
                "At least one of policy_id, customer_id, or customer_name "
                "is required to look up a policy."
            )
        }

    # Build query params from whichever identifiers were provided
    params: dict[str, str] = {}
    if policy_id:
        params["policy_id"] = policy_id
    elif customer_id:
        params["customer_id"] = customer_id
    elif customer_name:
        params["customer_name"] = customer_name

    log.info("crm.policy_lookup_started", params=params)

    try:
        data = await _fetch_policy(params)
        log.info(
            "crm.policy_lookup_success",
            policy_id=data.get("policy_id"),
            status=data.get("status"),
        )
        return data

    except httpx.HTTPStatusError as exc:
        # 404 = policy not found — tell the agent clearly
        if exc.response.status_code == 404:
            log.info("crm.policy_not_found", params=params)
            return {"error": "No policy found matching the provided details."}

        # Other HTTP errors — CRM returned an error response
        log.error(
            "crm.policy_lookup_http_error",
            status_code=exc.response.status_code,
            params=params,
        )
        return {"error": f"CRM error {exc.response.status_code}: could not retrieve policy."}

    except httpx.TimeoutException:
        log.error("crm.policy_lookup_timeout", params=params)
        return {"error": "Policy lookup timed out. Please try again."}

    except Exception as exc:
        log.error("crm.policy_lookup_failed", error=str(exc), params=params)
        return {"error": f"Policy lookup failed: {exc}"}