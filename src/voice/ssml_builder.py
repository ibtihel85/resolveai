"""
src/voice/ssml_builder.py

SSML builder — converts plain agent response text into SSML markup
for natural, accurate text-to-speech output.

Handles insurance-specific data types that generic TTS engines
mispronounce without guidance:
    - Policy IDs (POL-XXXXXXX) → spelled out digit by digit
    - Claim IDs  (CLM-XXXXXXX) → spelled out digit by digit
    - Currency amounts          → cardinal number + currency word
    - Dates                     → natural spoken date format
    - Insurance jargon          → phoneme overrides for accuracy
    - Confirmations             → emphasis for clarity
    - Transitions               → natural pauses for pacing

Two voice personas:
    "professional" — neutral pace, standard pitch (default)
    "warm"         — slightly slower, lower pitch, more empathetic
"""

from __future__ import annotations

import re
import xml.sax.saxutils as saxutils

# ── Pronunciation dictionary ──────────────────────────────────────────────────
# Maps plain text → IPA phoneme string.
# Used for insurance terms that TTS engines frequently mispronounce.
# IPA reference: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet
PRONUNCIATION_DICT: dict[str, tuple[str, str]] = {
    # word → (ipa_pronunciation, display_text)
    "Meridian":     ("məˈrɪdiən",    "Meridian"),
    "deductible":   ("dɪˈdʌktɪbəl", "deductible"),
    "coinsurance":  ("koʊɪnˈʃʊərəns", "coinsurance"),
    "subrogation":  ("ˌsʌbrəˈɡeɪʃən", "subrogation"),
    "beneficiary":  ("ˌbɛnɪˈfɪʃiɛri", "beneficiary"),
    "indemnity":    ("ɪnˈdɛmnɪti",   "indemnity"),
    "endorsement":  ("ɪnˈdɔːrsmənt", "endorsement"),
    "premium":      ("ˈpriːmiəm",    "premium"),
    "liability":    ("ˌlaɪəˈbɪlɪti", "liability"),
    "annuity":      ("əˈnjuːɪti",    "annuity"),
}

# ── Regex patterns ────────────────────────────────────────────────────────────
# Policy IDs: POL-XXXXXXX
_POLICY_ID_RE = re.compile(r"\b(POL-\d{5,})\b")

# Claim IDs: CLM-XXXXXXX
_CLAIM_ID_RE = re.compile(r"\b(CLM-\d{5,})\b")

# Currency: €500, €1,200, EUR 500, EUR500
_CURRENCY_RE = re.compile(
    r"(€|EUR\s?)([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Dates: 2026-07-15 (ISO format from CRM responses)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Confirmation phrases that deserve emphasis
_CONFIRMATION_RE = re.compile(
    r"\b(I've (created|booked|scheduled|cancelled|updated)|"
    r"Your (ticket|appointment|callback|policy|claim) (has been|is now))\b",
    re.IGNORECASE,
)

# Month names for date conversion
_MONTHS = {
    "01": "January", "02": "February", "03": "March",
    "04": "April",   "05": "May",      "06": "June",
    "07": "July",    "08": "August",   "09": "September",
    "10": "October", "11": "November", "12": "December",
}

# Persona settings
_PERSONAS = {
    "professional": {"rate": "100%", "pitch": "0%"},
    "warm":         {"rate": "92%",  "pitch": "-2%"},
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_ssml(text: str, persona: str = "professional") -> str:
    """
    Convert plain agent response text to SSML.

    Args:
        text:    Plain text agent response.
        persona: "professional" (default) or "warm".
                 Controls speaking rate and pitch.

    Returns:
        Complete SSML string ready for ElevenLabs or Azure Speech.
    """
    settings = _PERSONAS.get(persona, _PERSONAS["professional"])

    # Step 1 — XML-escape the raw text first to avoid injection
    text = saxutils.escape(text)

    # Step 2 — Apply transformations in order
    # Order matters: IDs before general numbers, currency before plain digits
    text = _transform_policy_ids(text)
    text = _transform_claim_ids(text)
    text = _transform_currency(text)
    text = _transform_dates(text)
    text = _transform_pronunciations(text)
    text = _transform_confirmations(text)

    # Step 3 — Wrap in SSML structure
    ssml = (
        f'<speak>\n'
        f'  <prosody rate="{settings["rate"]}" pitch="{settings["pitch"]}">\n'
        f'    {text}\n'
        f'  </prosody>\n'
        f'</speak>'
    )

    return ssml


def strip_ssml(ssml: str) -> str:
    """
    Remove all SSML tags and return plain text.
    Useful for logging the spoken content without markup.
    """
    return re.sub(r"<[^>]+>", "", ssml).strip()


# ── Private transformers ──────────────────────────────────────────────────────

def _transform_policy_ids(text: str) -> str:
    """
    POL-0023412 → P-O-L <break/> 0-0-2-3-4-1-2
    Spelled letter by letter with a pause after the prefix.
    """
    def replace(match: re.Match) -> str:
        raw = match.group(1)             # e.g. "POL-0023412"
        prefix, digits = raw.split("-", 1)
        # Spell prefix letters individually
        spelled_prefix = "-".join(prefix)
        # Spell digits individually
        spelled_digits = "-".join(digits)
        return (
            f'<say-as interpret-as="verbatim">{spelled_prefix}</say-as>'
            f'<break time="150ms"/>'
            f'<say-as interpret-as="verbatim">{spelled_digits}</say-as>'
            f'<break time="200ms"/>'
        )
    return _POLICY_ID_RE.sub(replace, text)


def _transform_claim_ids(text: str) -> str:
    """
    CLM-0012345 → C-L-M <break/> 0-0-1-2-3-4-5
    """
    def replace(match: re.Match) -> str:
        raw = match.group(1)
        prefix, digits = raw.split("-", 1)
        spelled_prefix = "-".join(prefix)
        spelled_digits = "-".join(digits)
        return (
            f'<say-as interpret-as="verbatim">{spelled_prefix}</say-as>'
            f'<break time="150ms"/>'
            f'<say-as interpret-as="verbatim">{spelled_digits}</say-as>'
            f'<break time="200ms"/>'
        )
    return _CLAIM_ID_RE.sub(replace, text)


def _transform_currency(text: str) -> str:
    """
    €500 → five hundred euros
    €1,200.00 → twelve hundred euros
    """
    def replace(match: re.Match) -> str:
        amount = match.group(2).replace(",", "")  # remove thousand separator
        return (
            f'<say-as interpret-as="cardinal">{amount}</say-as> euros'
        )
    return _CURRENCY_RE.sub(replace, text)


def _transform_dates(text: str) -> str:
    """
    2026-07-15 → July fifteenth, twenty twenty-six
    """
    def replace(match: re.Match) -> str:
        year = match.group(1)
        month = _MONTHS.get(match.group(2), match.group(2))
        day = match.group(3).lstrip("0") or "0"
        return (
            f'{month} '
            f'<say-as interpret-as="ordinal">{day}</say-as>, '
            f'<say-as interpret-as="cardinal">{year}</say-as>'
        )
    return _ISO_DATE_RE.sub(replace, text)


def _transform_pronunciations(text: str) -> str:
    """
    Apply phoneme overrides for insurance jargon.
    Case-sensitive matching — only overrides exact capitalisation.
    """
    for word, (ipa, display) in PRONUNCIATION_DICT.items():
        phoneme_tag = (
            f'<phoneme alphabet="ipa" ph="{ipa}">{display}</phoneme>'
        )
        # Replace whole-word occurrences only
        text = re.sub(
            rf"\b{re.escape(word)}\b",
            phoneme_tag,
            text,
        )
    return text


def _transform_confirmations(text: str) -> str:
    """
    Add emphasis to confirmation phrases for clarity.
    "I've created ticket #12345" → emphasized
    """
    return _CONFIRMATION_RE.sub(
        lambda m: f'<emphasis level="moderate">{m.group(0)}</emphasis>',
        text,
    )