"""
src/agent/guardrails.py

Guardrails layer — input validation, PII redaction, and escalation triggers.

Applied before every LLM call in src/agent/core.py.

Input guards (block before LLM sees the message):
    - Prompt injection detection
    - Out-of-scope topic detection
    - Legal/financial/medical advice requests

PII handling:
    - Original message sent to LLM unchanged (customer needs to share info)
    - Redacted version stored in logs (GDPR compliance)

Escalation triggers (checked independently of LLM judgment):
    - Explicit human request
    - Anger/distress signals
    - Low retrieval confidence
    - Repeated tool failures
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.logger import get_logger

log = get_logger(__name__)

# ── Presidio setup ────────────────────────────────────────────────────────────
# Instantiated once at module load — loading NLP models is expensive.
_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# ── Injection patterns ────────────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+your\s+(system\s+)?prompt",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+)?",
    r"jailbreak",
    r"dan\s+mode",
    r"<\s*system\s*>",
    r"disregard\s+(your\s+)?(previous\s+)?instructions?",
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE,
)

# ── Out-of-scope patterns ─────────────────────────────────────────────────────
_OUT_OF_SCOPE_PATTERNS = [
    r"\b(recipe|cook(ing)?|restaurant)\b",
    r"\b(weather|forecast|temperature)\b",
    r"\b(sport(s)?|football|soccer|tennis)\b",
    r"\b(movie|film|series|netflix)\b",
    r"\b(stock\s+price|crypto|bitcoin|invest(ment)?)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay|code)\b",
    r"\btranslate\b",
]
_OUT_OF_SCOPE_RE = re.compile(
    "|".join(_OUT_OF_SCOPE_PATTERNS),
    re.IGNORECASE,
)

# ── Advice request patterns ───────────────────────────────────────────────────
_ADVICE_PATTERNS = [
    r"\b(should\s+I\s+sue|file\s+a\s+lawsuit|legal\s+action)\b",
    r"\bis\s+(this|it)\s+legal\b",
    r"\b(tax\s+advice|tax\s+return|financial\s+advice)\b",
    r"\b(medical\s+advice|diagnos(e|is)|symptoms?)\b",
]
_ADVICE_RE = re.compile(
    "|".join(_ADVICE_PATTERNS),
    re.IGNORECASE,
)

# ── Escalation patterns ───────────────────────────────────────────────────────
_HUMAN_REQUEST_RE = re.compile(
    r"\b(speak|talk|connect|transfer|escalate)\s+(to|with)\s+"
    r"(a\s+)?(real\s+)?(human|person|agent|representative|someone)\b",
    re.IGNORECASE,
)

_ANGER_RE = re.compile(
    r"\b(furious|outraged|unacceptable|disgusting|terrible|awful|"
    r"worst|useless|incompetent|ridiculous|lawsuit|sue\s+you)\b",
    re.IGNORECASE,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    Result of a guardrail check.

    passed:           True if the input is allowed through
    triggered_rule:   Name of the rule that fired, if any
    block_reason:     Human-readable reason shown to the customer
    redacted_text:    PII-scrubbed version for logging (None if no PII found)
    """
    passed: bool
    triggered_rule: str | None = None
    block_reason: str | None = None
    redacted_text: str | None = None


# ── Guardrails layer ──────────────────────────────────────────────────────────

class GuardrailsLayer:
    """
    Stateless guardrail checks — instantiate once, call per turn.

    All methods are synchronous — guardrail checks must be fast.
    No external calls, no LLM, no I/O.
    """

    def check_input(self, text: str) -> GuardrailResult:
        """
        Run all input-side guardrail checks in priority order.

        Returns on the first match — checks are ordered from most
        to least severe.
        """

        # 1. Prompt injection — highest priority
        if _INJECTION_RE.search(text):
            log.warning("guardrail.injection_detected", preview=text[:80])
            return GuardrailResult(
                passed=False,
                triggered_rule="prompt_injection",
                block_reason=(
                    "I'm not able to process that request. "
                    "Is there something I can help you with regarding "
                    "your Meridian Insurance policy?"
                ),
            )

        # 2. Out-of-scope topic
        if _OUT_OF_SCOPE_RE.search(text):
            log.info("guardrail.out_of_scope", preview=text[:80])
            return GuardrailResult(
                passed=False,
                triggered_rule="out_of_scope",
                block_reason=(
                    "That's outside what I can help with here. "
                    "I specialise in policy questions, claims, billing, "
                    "and appointments. Can I help you with any of those?"
                ),
            )

        # 3. Legal/financial/medical advice
        if _ADVICE_RE.search(text):
            log.info("guardrail.advice_request", preview=text[:80])
            return GuardrailResult(
                passed=False,
                triggered_rule="advice_request",
                block_reason=(
                    "I'm not able to provide legal, financial, or medical advice. "
                    "For those questions, I'd recommend speaking with "
                    "a qualified professional."
                ),
            )

        # 4. PII detection — input passes but log the redacted version
        redacted = self._redact_pii(text)
        if redacted != text:
            log.info("guardrail.pii_detected_in_input")
            return GuardrailResult(
                passed=True,
                triggered_rule="pii_in_input",
                redacted_text=redacted,
            )

        return GuardrailResult(passed=True)

    def should_escalate(
        self,
        text: str,
        retrieval_score: float | None,
        retry_count: int,
    ) -> tuple[bool, str | None]:
        """
        Determine whether this turn should trigger human escalation.

        Returns (should_escalate, reason_string).
        Called by core.py before the LLM call — escalation decisions
        are made by deterministic rules, not by the LLM itself.
        """

        # Explicit request for a human
        if _HUMAN_REQUEST_RE.search(text):
            return True, "customer_requested_human"

        # Anger or distress signals
        if _ANGER_RE.search(text):
            return True, "anger_detected"

        # Retrieval confidence too low to trust an answer
        if (
            retrieval_score is not None
            and retrieval_score < 0.35
        ):
            return True, "low_retrieval_confidence"

        # Too many retries — agent is stuck
        if retry_count >= 2:
            return True, "repeated_tool_failure"

        return False, None

    def _redact_pii(self, text: str) -> str:
        """
        Replace PII in text with placeholders for safe logging.
        Returns the original text unchanged if no PII is detected.
        """
        try:
            results = _analyzer.analyze(text=text, language="en")
            if not results:
                return text
            anonymized = _anonymizer.anonymize(
                text=text,
                analyzer_results=results,
            )
            return anonymized.text
        except Exception as exc:
            # PII redaction failure must never block the conversation
            log.warning("guardrail.pii_redaction_failed", error=str(exc))
            return text