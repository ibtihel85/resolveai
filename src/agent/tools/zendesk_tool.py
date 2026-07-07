"""
src/agent/tools/zendesk_tool.py

Zendesk ticketing tool — creates support tickets via the Zendesk REST API.

Used for:
    - Escalating conversations to human agents
    - Creating tickets for unresolvable customer requests
    - Logging customer issues that require follow-up

Authentication:
    Zendesk uses HTTP Basic Auth with email/token format.
    Set ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN in .env.
    Get a free sandbox at: https://www.zendesk.com/register/free-trial

Graceful degradation:
    If credentials are not configured, returns a mock ticket ID
    so the rest of the escalation flow works during local development.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Tool definition ───────────────────────────────────────────────────────────
TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": (
            "Create a Zendesk support ticket to escalate the conversation "
            "to a human agent or log an unresolvable customer request. "
            "Always call this when escalating — never escalate without "
            "creating a ticket first. "
            "Include a clear summary of the customer's issue and what "
            "has already been attempted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": (
                        "One-line summary of the issue (max 150 characters). "
                        "Example: 'Claim CLM-0012345 status dispute — customer escalation'"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Full description of the issue including: "
                        "what the customer asked, what was already tried, "
                        "and why escalation is needed."
                    ),
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": (
                        "Ticket priority. Use 'high' for angry customers "
                        "or time-sensitive issues. Use 'urgent' for "
                        "legal threats or safety concerns."
                    ),
                    "default": "normal",
                },
                "customer_name": {
                    "type": "string",
                    "description": "Customer name if known.",
                },
                "policy_id": {
                    "type": "string",
                    "description": "Policy ID if relevant to the issue.",
                },
            },
            "required": ["subject", "description"],
        },
    },
}


# ── Auth helper ───────────────────────────────────────────────────────────────

def _build_auth_header() -> str:
    """
    Build the Zendesk Basic Auth header value.
    Format: Base64(email/token:api_token)
    """
    credentials = f"{settings.zendesk_email}/token:{settings.zendesk_api_token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def _is_configured() -> bool:
    """Return True if Zendesk credentials are set in .env."""
    return bool(
        settings.zendesk_subdomain
        and settings.zendesk_email
        and settings.zendesk_api_token
    )


# ── Tool handler ──────────────────────────────────────────────────────────────

async def run(
    subject: str,
    description: str,
    priority: str = "normal",
    customer_name: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a Zendesk support ticket.

    Returns ticket_id on success.
    Returns a mock ticket if credentials are not configured.
    Never raises — errors are returned as dicts.
    """

    # Graceful degradation for local dev without Zendesk credentials
    if not _is_configured():
        log.warning("zendesk.not_configured_using_mock")
        return {
            "ticket_id": "MOCK-001",
            "status": "created",
            "url": "https://mock.zendesk.com/agent/tickets/1",
            "note": "Zendesk not configured — mock ticket created.",
        }

    # Build ticket payload
    # Append context fields to description for agent visibility
    full_description = description
    if customer_name:
        full_description += f"\n\nCustomer: {customer_name}"
    if policy_id:
        full_description += f"\nPolicy ID: {policy_id}"

    payload = {
        "ticket": {
            "subject": subject[:150],   # Zendesk subject limit
            "comment": {"body": full_description},
            "priority": priority,
            "tags": ["resolveai", "ai-escalation"],
        }
    }

    url = f"https://{settings.zendesk_subdomain}.zendesk.com/api/v2/tickets.json"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": _build_auth_header(),
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            ticket = response.json()["ticket"]
            ticket_id = str(ticket["id"])

            log.info(
                "zendesk.ticket_created",
                ticket_id=ticket_id,
                priority=priority,
                subject=subject[:60],
            )

            return {
                "ticket_id": ticket_id,
                "status": ticket["status"],
                "url": (
                    f"https://{settings.zendesk_subdomain}.zendesk.com"
                    f"/agent/tickets/{ticket_id}"
                ),
            }

    except httpx.HTTPStatusError as exc:
        log.error(
            "zendesk.http_error",
            status_code=exc.response.status_code,
            body=exc.response.text[:200],
        )
        return {
            "error": (
                f"Zendesk returned {exc.response.status_code}. "
                "Ticket could not be created."
            )
        }

    except httpx.TimeoutException:
        log.error("zendesk.timeout")
        return {"error": "Zendesk request timed out. Ticket could not be created."}

    except Exception as exc:
        log.error("zendesk.unexpected_error", error=str(exc))
        return {"error": f"Unexpected error creating ticket: {exc}"}