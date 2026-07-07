"""
src/agent/tools/claims_tool.py

Claims status tool — retrieves live claim status from the mock CRM.

Always use lookup_policy first if you don't have a policy_id.
Never use the knowledge base for claim status — it must come from
the authoritative CRM system.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_claim_status",
        "description": (
            "Retrieve the current status of an insurance claim from the "
            "Meridian Insurance system. "
            "Use this when a customer asks about a specific claim. "
            "Requires either claim_id or policy_id. "
            "Call lookup_policy first if you only have the customer name. "
            "Returns claim status, date filed, estimated resolution date, "
            "amount claimed, and adjuster notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {
                    "type": "string",
                    "description": (
                        "Claim identifier, format CLM-XXXXXXX. "
                        "Use this if the customer provides a claim number directly."
                    ),
                },
                "policy_id": {
                    "type": "string",
                    "description": (
                        "Policy ID to retrieve all claims for that policy. "
                        "Use when claim_id is unknown but policy_id is available."
                    ),
                },
            },
        },
    },
}


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    reraise=True,
)
async def _fetch_claims(params: dict[str, str]) -> dict[str, Any] | list:
    """Internal HTTP call to the mock CRM — wrapped in retry logic."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(
            f"{settings.mock_crm_url}/v1/claims",
            params=params,
            headers={"X-API-Key": settings.mock_crm_api_key},
        )
        response.raise_for_status()
        return response.json()


async def run(
    claim_id: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any] | list:
    """
    Retrieve claim status from the mock CRM.

    Returns claim data on success, or {"error": "..."} on failure.
    Never raises — errors are returned as dicts.
    """
    if not any([claim_id, policy_id]):
        return {
            "error": (
                "At least one of claim_id or policy_id is required "
                "to look up a claim."
            )
        }

    params: dict[str, str] = {}
    if claim_id:
        params["claim_id"] = claim_id
    elif policy_id:
        params["policy_id"] = policy_id

    log.info("crm.claim_lookup_started", params=params)

    try:
        data = await _fetch_claims(params)
        log.info("crm.claim_lookup_success", params=params)
        return data

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            log.info("crm.claim_not_found", params=params)
            return {"error": "No claim found matching the provided details."}
        log.error("crm.claim_lookup_http_error", status_code=exc.response.status_code)
        return {"error": f"CRM error {exc.response.status_code}."}

    except httpx.TimeoutException:
        log.error("crm.claim_lookup_timeout")
        return {"error": "Claim lookup timed out. Please try again."}

    except Exception as exc:
        log.error("crm.claim_lookup_failed", error=str(exc))
        return {"error": f"Claim lookup failed: {exc}"}