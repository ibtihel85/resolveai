"""
src/agent/tools/slack_tool.py

Slack escalation notification tool — posts a rich alert to the
support team channel when a conversation is escalated.

Called alongside create_ticket during every escalation.
Gives the support team immediate visibility without checking Zendesk.

Setup:
    1. Create a Slack app with Incoming Webhooks enabled
    2. Add a webhook for your #support-escalations channel
    3. Set SLACK_WEBHOOK_URL in .env

Graceful degradation:
    If SLACK_WEBHOOK_URL is not configured, logs a warning and
    returns success — escalation completes without Slack notification.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Tool definition ───────────────────────────────────────────────────────────
TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "notify_slack_escalation",
        "description": (
            "Send an escalation alert to the support team Slack channel. "
            "Call this alongside create_ticket whenever escalating to a human agent. "
            "Provides the support team with immediate notification and full context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the conversation is being escalated.",
                },
                "customer_name": {
                    "type": "string",
                    "description": "Customer name if known.",
                },
                "policy_id": {
                    "type": "string",
                    "description": "Policy ID if relevant.",
                },
                "ticket_id": {
                    "type": "string",
                    "description": "Zendesk ticket ID from create_ticket call.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Internal conversation ID for audit trail.",
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Brief 1-2 sentence summary of the conversation "
                        "and why escalation was needed."
                    ),
                },
            },
            "required": ["reason", "conversation_id"],
        },
    },
}


# ── Tool handler ──────────────────────────────────────────────────────────────

async def run(
    reason: str,
    conversation_id: str,
    customer_name: str | None = None,
    policy_id: str | None = None,
    ticket_id: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """
    Post an escalation alert to Slack via Incoming Webhook.

    Returns success dict on success or if Slack is not configured.
    Never raises — Slack notification failure must not affect escalation.
    """

    if not settings.slack_webhook_url:
        log.warning("slack.not_configured")
        return {"status": "skipped", "reason": "SLACK_WEBHOOK_URL not configured"}

    # ── Build Block Kit message ───────────────────────────────────────────────
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🔔 ResolveAI — Escalation Required",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Reason:*\n{reason}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Conversation ID:*\n`{conversation_id}`",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Customer:*\n{customer_name or '—'}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Policy ID:*\n{policy_id or '—'}",
                },
            ],
        },
    ]

    # Add ticket link if available
    if ticket_id and settings.zendesk_subdomain:
        ticket_url = (
            f"https://{settings.zendesk_subdomain}.zendesk.com"
            f"/agent/tickets/{ticket_id}"
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Zendesk Ticket:* <{ticket_url}|#{ticket_id}>",
            },
        })
    elif ticket_id:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Zendesk Ticket:* #{ticket_id}",
            },
        })

    # Add conversation summary if provided
    if summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary:*\n{summary}",
            },
        })

    # Divider at the end
    blocks.append({"type": "divider"})

    payload = {
        "text": f"Escalation: {reason} — {customer_name or 'Unknown customer'}",
        "blocks": blocks,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                settings.slack_webhook_url,
                json=payload,
            )
            response.raise_for_status()

            log.info(
                "slack.escalation_sent",
                conversation_id=conversation_id,
                reason=reason,
                ticket_id=ticket_id,
            )
            return {"status": "sent", "channel": settings.slack_escalation_channel}

    except httpx.HTTPStatusError as exc:
        log.error(
            "slack.http_error",
            status_code=exc.response.status_code,
            body=exc.response.text[:200],
        )
        return {"error": f"Slack returned {exc.response.status_code}"}

    except httpx.TimeoutException:
        log.error("slack.timeout")
        return {"error": "Slack request timed out"}

    except Exception as exc:
        log.error("slack.unexpected_error", error=str(exc))
        return {"error": f"Unexpected error sending Slack notification: {exc}"}