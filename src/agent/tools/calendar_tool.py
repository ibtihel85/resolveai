"""
src/agent/tools/calendar_tool.py

Google Calendar booking tool — schedules callback appointments
between customers and Meridian Insurance agents.

Authentication:
    Uses OAuth 2.0 credentials stored in credentials/google_credentials.json
    Run scripts/setup_google_auth.py once to complete the OAuth flow.

Graceful degradation:
    If credentials are not configured, returns a mock confirmation
    so the conversation flow completes during local development.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Tool definition ───────────────────────────────────────────────────────────
TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_callback",
        "description": (
            "Schedule a callback appointment for the customer with a "
            "Meridian Insurance agent. "
            "Use when the customer requests to speak with someone at a "
            "specific time, or when offering a callback as part of escalation. "
            "Returns a confirmation with the event details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Full name of the customer.",
                },
                "customer_email": {
                    "type": "string",
                    "description": "Customer email address for the calendar invite.",
                },
                "preferred_date": {
                    "type": "string",
                    "description": (
                        "Preferred date in YYYY-MM-DD format. "
                        "Example: '2025-06-15'"
                    ),
                },
                "preferred_time": {
                    "type": "string",
                    "description": (
                        "Preferred time in HH:MM format (24-hour). "
                        "Example: '14:00'"
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Brief description of what the callback is about. "
                        "Example: 'Policy renewal discussion'"
                    ),
                },
                "policy_id": {
                    "type": "string",
                    "description": "Policy ID if relevant to the callback topic.",
                },
            },
            "required": ["customer_name", "preferred_date", "preferred_time", "topic"],
        },
    },
}


# ── Credential helpers ────────────────────────────────────────────────────────

def _credentials_exist() -> bool:
    """Return True if Google credentials file exists on disk."""
    return Path(settings.google_credentials_file).exists()


def _load_credentials():
    """
    Load Google OAuth credentials from disk.
    Returns None if credentials are not available.
    """
    try:
        from google.oauth2.credentials import Credentials

        creds_path = Path(settings.google_credentials_file)
        if not creds_path.exists():
            return None

        import json
        cred_data = json.loads(creds_path.read_text())

        # Check if this is an authorized user credential (after OAuth flow)
        if "token" in cred_data:
            return Credentials.from_authorized_user_info(
                cred_data,
                scopes=["https://www.googleapis.com/auth/calendar.events"],
            )
        return None

    except Exception as exc:
        log.warning("calendar.credentials_load_failed", error=str(exc))
        return None


# ── Tool handler ──────────────────────────────────────────────────────────────

async def run(
    customer_name: str,
    preferred_date: str,
    preferred_time: str,
    topic: str,
    customer_email: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a Google Calendar event for a customer callback.

    Returns event details on success.
    Returns mock confirmation if credentials are not configured.
    Never raises — errors are returned as dicts.
    """

    # Parse date and time
    try:
        start_dt = datetime.strptime(
            f"{preferred_date} {preferred_time}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError:
        return {
            "error": (
                f"Invalid date or time format. "
                f"Use YYYY-MM-DD for date and HH:MM for time. "
                f"Received: date='{preferred_date}', time='{preferred_time}'"
            )
        }

    end_dt = start_dt + timedelta(minutes=30)

    # Build event title and description
    title = f"Callback — {customer_name} — {topic}"
    description_parts = [
        f"Customer: {customer_name}",
        f"Topic: {topic}",
    ]
    if policy_id:
        description_parts.append(f"Policy ID: {policy_id}")
    if customer_email:
        description_parts.append(f"Email: {customer_email}")
    description = "\n".join(description_parts)

    # Graceful degradation — return mock if credentials not configured
    creds = _load_credentials()
    if creds is None:
        log.warning("calendar.not_configured_using_mock")
        return {
            "event_id": "MOCK-EVENT-001",
            "status": "confirmed",
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "note": "Google Calendar not configured — mock booking created.",
        }

    # ── Create real Google Calendar event ────────────────────────────────────
    try:
        from googleapiclient.discovery import build

        service = build("calendar", "v3", credentials=creds)

        event: dict[str, Any] = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": "Europe/Berlin",
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": "Europe/Berlin",
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 15},
                ],
            },
        }

        # Add customer as attendee if email provided
        if customer_email:
            event["attendees"] = [{"email": customer_email}]

        created_event = (
            service.events()
            .insert(
                calendarId=settings.google_calendar_id,
                body=event,
                sendUpdates="all" if customer_email else "none",
            )
            .execute()
        )

        log.info(
            "calendar.event_created",
            event_id=created_event["id"],
            start=start_dt.isoformat(),
            customer=customer_name,
        )

        return {
            "event_id": created_event["id"],
            "status": created_event["status"],
            "title": created_event["summary"],
            "start": created_event["start"]["dateTime"],
            "end": created_event["end"]["dateTime"],
            "html_link": created_event.get("htmlLink"),
        }

    except Exception as exc:
        log.error("calendar.create_failed", error=str(exc))
        return {"error": f"Failed to create calendar event: {exc}"}