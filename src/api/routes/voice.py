"""
src/api/routes/voice.py

Voice channel routes — handles Twilio webhooks for inbound phone calls.

Call flow:
    1. POST /v1/voice/incoming  — Twilio calls this when a customer dials in.
                                  Returns TwiML greeting + Gather instruction.

    2. POST /v1/voice/respond   — Twilio calls this after customer speaks.
                                  Receives audio, transcribes, runs agent,
                                  synthesizes response, returns TwiML.

    3. POST /v1/voice/status    — Twilio calls this when call ends.
                                  Cleans up the session.

Session management:
    Voice sessions are keyed by Twilio's CallSid — a unique ID per call.
    One ConversationManager per call, same as chat sessions.

TwiML responses:
    We use Twilio's <Say> with our synthesized audio URL for voice output,
    or fall back to Twilio's built-in TTS if audio generation fails.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request, Response

from src.agent.core import ConversationManager
from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

# ── Voice session store ───────────────────────────────────────────────────────
# Keyed by Twilio CallSid — one ConversationManager per active call.
_voice_sessions: dict[str, ConversationManager] = {}


# ── TwiML helpers ─────────────────────────────────────────────────────────────

def _twiml_gather(say_text: str, action_url: str) -> str:
    """
    Build TwiML that speaks text then listens for customer speech.
    The customer's speech is sent to action_url when they stop speaking.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna-Neural">{say_text}</Say>
    <Gather
        input="speech"
        action="{action_url}"
        method="POST"
        speechTimeout="auto"
        language="en-US"
        actionOnEmptyResult="true">
    </Gather>
</Response>"""


def _twiml_say_and_hangup(say_text: str) -> str:
    """
    Build TwiML that speaks text then ends the call.
    Used for escalation farewells.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna-Neural">{say_text}</Say>
    <Hangup/>
</Response>"""


def _twiml_no_input(action_url: str) -> str:
    """
    Build TwiML for when no speech was detected.
    Prompts the customer to speak again.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna-Neural">
        I didn't catch that. Could you please repeat your question?
    </Say>
    <Gather
        input="speech"
        action="{action_url}"
        method="POST"
        speechTimeout="auto"
        language="en-US"
        actionOnEmptyResult="true">
    </Gather>
</Response>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/incoming")
async def voice_incoming(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(default="unknown"),
) -> Response:
    """
    Twilio calls this endpoint when a customer dials the Meridian Insurance
    phone number.

    Returns TwiML that:
    1. Greets the customer as Aria
    2. Listens for their first question
    """
    log.info(
        "voice.call_incoming",
        call_sid=CallSid,
        from_number=From,
    )

    # Create a new conversation session for this call
    manager = ConversationManager(
        conversation_id=CallSid,
        channel="voice",
    )
    _voice_sessions[CallSid] = manager

    respond_url = f"{settings.public_base_url}/v1/voice/respond"

    greeting = (
        "Thank you for calling Meridian Insurance. "
        "I'm Aria, your AI assistant. "
        "How can I help you today?"
    )

    twiml = _twiml_gather(greeting, respond_url)
    return Response(content=twiml, media_type="application/xml")


@router.post("/respond")
async def voice_respond(
    request: Request,
    CallSid: str = Form(...),
    SpeechResult: str = Form(default=""),
    Confidence: float = Form(default=0.0),
) -> Response:
    """
    Twilio calls this endpoint after the customer finishes speaking.

    Receives the transcribed speech (Twilio does basic STT),
    runs the agent, synthesizes the response, returns TwiML.

    Note: We use Twilio's built-in speech recognition (SpeechResult)
    rather than our Whisper STT for simplicity in the webhook flow.
    For higher accuracy, replace SpeechResult with audio streaming + Whisper.
    """
    log.info(
        "voice.speech_received",
        call_sid=CallSid,
        speech_result=SpeechResult[:80],
        confidence=Confidence,
    )

    respond_url = f"{settings.public_base_url}/v1/voice/respond"

    # Handle empty speech
    if not SpeechResult.strip():
        log.info("voice.no_speech_detected", call_sid=CallSid)
        twiml = _twiml_no_input(respond_url)
        return Response(content=twiml, media_type="application/xml")

    # Get or create session
    manager = _voice_sessions.get(CallSid)
    if not manager:
        log.warning("voice.session_not_found_creating_new", call_sid=CallSid)
        manager = ConversationManager(
            conversation_id=CallSid,
            channel="voice",
        )
        _voice_sessions[CallSid] = manager

    # Run the agent
    result = await manager.handle_turn(SpeechResult.strip())

    response_text = result.response_text

    # If escalated — say farewell and hang up
    if result.is_escalation:
        _voice_sessions.pop(CallSid, None)
        farewell = (
            f"{response_text} "
            "Thank you for calling Meridian Insurance. Goodbye."
        )
        twiml = _twiml_say_and_hangup(farewell)
        log.info("voice.call_escalated", call_sid=CallSid)
        return Response(content=twiml, media_type="application/xml")

    # Continue the conversation
    twiml = _twiml_gather(response_text, respond_url)
    return Response(content=twiml, media_type="application/xml")


@router.post("/status")
async def voice_status(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
) -> dict[str, Any]:
    """
    Twilio status callback — called when call state changes.
    We use it to clean up sessions when calls end.
    """
    log.info(
        "voice.status_update",
        call_sid=CallSid,
        status=CallStatus,
    )

    if CallStatus in ("completed", "failed", "busy", "no-answer", "canceled"):
        manager = _voice_sessions.pop(CallSid, None)
        if manager:
            log.info(
                "voice.session_cleaned",
                call_sid=CallSid,
                turns=manager.memory.turn_count(),
            )

    return {"status": "ok"}


@router.get("/health")
async def voice_health() -> dict[str, Any]:
    """Voice channel health check — shows active call count."""
    return {
        "status": "ok",
        "active_calls": len(_voice_sessions),
        "public_base_url": settings.public_base_url,
    }