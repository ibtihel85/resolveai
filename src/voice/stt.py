"""
src/voice/stt.py

Speech-to-text — transcribes customer audio to text using
OpenAI Whisper running locally.

Audio arrives from Twilio as raw bytes (WAV or MP3).
Whisper converts it to text which is passed to the agent loop.

The agent loop (core.py) is identical for voice and chat —
STT is the only voice-specific preprocessing step on the input side.

Model sizes (tradeoff: accuracy vs speed vs disk):
    tiny   — fastest, least accurate (~75MB)
    base   — good balance for English (~140MB)  ← default
    small  — better accuracy (~460MB)
    medium — near human-level (~1.5GB)
    large  — best accuracy (~3GB)
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Any

from src.logger import get_logger

log = get_logger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
# "base" is the default — good accuracy for English, downloads once (~140MB)
WHISPER_MODEL = "base"


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """
    Load the Whisper model once and cache it.
    First call downloads the model if not already cached locally.
    Subsequent calls return the cached model instantly.
    """
    import whisper

    log.info("stt.model_loading", model=WHISPER_MODEL)
    model = whisper.load_model(WHISPER_MODEL)
    log.info("stt.model_ready", model=WHISPER_MODEL)
    return model


# ── Public API ────────────────────────────────────────────────────────────────

async def transcribe(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    language: str = "en",
) -> str:
    """
    Transcribe audio bytes to text using local Whisper.

    Args:
        audio_bytes:  Raw audio data from Twilio.
        content_type: MIME type of the audio (default: audio/wav).
        language:     Language code (default: "en").
                      Pass "de" for German, "fr" for French.

    Returns:
        Transcribed text string.
    """
    log.info(
        "stt.transcription_started",
        audio_bytes=len(audio_bytes),
        content_type=content_type,
    )

    # Whisper needs a file path, not bytes — write to temp file
    suffix = ".wav"
    if "mp3" in content_type:
        suffix = ".mp3"
    elif "ogg" in content_type:
        suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _load_model()
        result = model.transcribe(
            tmp_path,
            language=language,
            fp16=False,   # fp16=False for CPU inference (no GPU needed)
        )
        text = result["text"].strip()

        log.info(
            "stt.transcription_complete",
            text_length=len(text),
            preview=text[:60],
        )
        return text

    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


async def transcribe_with_fallback(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    language: str = "en",
    fallback_text: str = "",
) -> str:
    """
    Transcribe audio with graceful fallback.

    Returns fallback_text if transcription fails — allows the voice
    flow to continue rather than crashing the Twilio webhook.
    """
    try:
        return await transcribe(audio_bytes, content_type, language)
    except Exception as exc:
        log.error(
            "stt.transcription_failed",
            error=str(exc),
            fallback=bool(fallback_text),
        )
        return fallback_text