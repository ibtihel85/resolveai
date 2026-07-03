"""
src/agent/prompts/__init__.py

Prompt registry — loads versioned system prompt files from disk,
caches them, strips YAML frontmatter, and injects runtime case context.

This is the only place in the codebase that reads prompt files.
src/agent/core.py calls get_system_prompt() on every LLM call.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from jinja2 import Template

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# Absolute path to the prompts directory
# Path(__file__) = .../src/agent/prompts/__init__.py
# .parent         = .../src/agent/prompts/
PROMPTS_DIR = Path(__file__).parent

# Regex that matches the YAML frontmatter block:
# everything between the opening --- and closing ---
_FRONTMATTER_RE = re.compile(r"^---\n.*?---\n", re.DOTALL)


@lru_cache(maxsize=8)
def _load_raw_prompt(version: str) -> str:
    """
    Load a prompt file from disk and strip its YAML frontmatter.
    Result is cached — disk is only read once per version per process.

    Args:
        version: prompt version string, e.g. "v1" or "v2"

    Returns:
        Clean prompt text with frontmatter removed, ready for Jinja2 rendering.

    Raises:
        FileNotFoundError: if no prompt file exists for this version.
    """
    path = PROMPTS_DIR / f"system_prompt_{version}.md"

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' not found at {path}. "
            f"Available versions: {list_versions()}"
        )

    raw = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter — the LLM does not need metadata
    clean = _FRONTMATTER_RE.sub("", raw).strip()

    log.info("prompt.loaded", version=version, chars=len(clean))
    return clean


def get_system_prompt(version: str, case_context: str) -> str:
    """
    Return the system prompt for a given version with case context injected.

    This is called by src/agent/core.py on every LLM call.

    Args:
        version:      prompt version string, e.g. "v1"
        case_context: structured string describing the current conversation
                      state — injected into {{ case_context }} in the template.

    Returns:
        Complete system prompt string ready to send to the LLM.
    """
    raw = _load_raw_prompt(version)

    # Use Jinja2 to replace {{ case_context }} with actual conversation state
    template = Template(raw)
    rendered = template.render(case_context=case_context)

    return rendered


def list_versions() -> list[str]:
    """
    Return all available prompt version strings, sorted.

    Example return value: ["v1", "v2", "v3"]

    Used by the eval harness and the /v1/eval/versions API endpoint.
    """
    return sorted(
        p.stem.replace("system_prompt_", "")
        for p in PROMPTS_DIR.glob("system_prompt_v*.md")
    )