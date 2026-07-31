"""
src/voice/tts.py

Text-to-speech — converts agent response text to audio bytes.

Two providers in priority order:
    1. ElevenLabs API (cloud) — high quality, natural voice, SSML support.
       Free tier: 10,000 characters/month. Set ELEVENLABS_API_KEY in .env.

    2. pyttsx3 (local fallback) — uses OS built-in speech engine.
       Free, offline, no API key. Lower quality but always available.

The SSML builder (ssml_builder.py) is applied before sending to ElevenLabs.
pyttsx3 does not support SSML — plain text is used for the local fallback.

Voice personas:
    "professional" — neutral, clear, business-appropriate
    "warm"         — slightly slower, more empathetic
"""

from __future__ import annotations

import asyncio

from src.config import settings
from src.logger import get_logger
from src.voice.ssml_builder import build_ssml, strip_ssml

log = get_logger(__name__)

# ── ElevenLabs voice IDs ──────────────────────────────────────────────────────
# Free voices available on all plans.
# Full list: https://api.elevenlabs.io/v1/voices
_VOICE_IDS = {
    "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel — clear, professional
    "warm":         "AZnzlk1XvdvUeBnXmlld",  # Domi — warm, empathetic
}

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


# ── Public API ────────────────────────────────────────────────────────────────

async def synthesize(
    text: str,
    persona: str = "professional",
) -> bytes:
    """
    Convert text to speech audio bytes.

    Tries ElevenLabs first (if configured), falls back to pyttsx3.

    Args:
        text:    Plain text agent response.
        persona: "professional" or "warm" — affects voice and SSML tuning.

    Returns:
        Audio bytes (MP3 for ElevenLabs, WAV for pyttsx3 fallback).
    """
    if settings.elevenlabs_api_key:
        try:
            return await _synthesize_elevenlabs(text, persona)
        except Exception as exc:
            log.warning(
                "tts.elevenlabs_failed_using_fallback",
                error=str(exc),
            )

    # Fallback to local pyttsx3
    return await _synthesize_local(text)


# ── ElevenLabs ────────────────────────────────────────────────────────────────

async def _synthesize_elevenlabs(text: str, persona: str) -> bytes:
    """
    Synthesize speech using ElevenLabs API with SSML.
    Returns MP3 audio bytes.
    """
    import httpx

    voice_id = _VOICE_IDS.get(persona, _VOICE_IDS["professional"])

    # Build SSML from plain text
    ssml_text = build_ssml(text, persona=persona)

    payload = {
        "text": ssml_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    headers = {
        "xi-api-key": settings.elevenlabs_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    url = f"{_ELEVENLABS_BASE}/text-to-speech/{voice_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

    log.info(
        "tts.elevenlabs_complete",
        persona=persona,
        text_length=len(text),
        audio_bytes=len(response.content),
    )

    return response.content


# ── Local pyttsx3 fallback ────────────────────────────────────────────────────

async def _synthesize_local(text: str) -> bytes:
    """
    Synthesize speech locally using pyttsx3.
    Returns WAV audio bytes.

    pyttsx3 is synchronous — we run it in a thread pool
    to avoid blocking the async event loop.
    """
    # Strip any SSML tags — pyttsx3 does not support SSML
    plain_text = strip_ssml(text) if "<" in text else text

    loop = asyncio.get_event_loop()
    audio_bytes = await loop.run_in_executor(
        None,
        _pyttsx3_synthesize_sync,
        plain_text,
    )

    log.info(
        "tts.local_complete",
        text_length=len(plain_text),
        audio_bytes=len(audio_bytes),
    )

    return audio_bytes


def _pyttsx3_synthesize_sync(text: str) -> bytes:
    """
    Synchronous pyttsx3 synthesis — runs in thread pool.
    Saves audio to a temp file and reads it back as bytes.
    """
    import os
    import tempfile

    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)    # words per minute
    engine.setProperty("volume", 0.9)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        engine.stop()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)