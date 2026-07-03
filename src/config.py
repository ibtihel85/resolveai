"""
src/config.py

Centralised settings for ResolveAI.
Loaded once at startup via pydantic-settings.

Every other module imports the `settings` object from here.
No other file calls os.getenv() directly.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    pydantic-settings reads values from environment variables or a .env file.
    Types are validated at startup — missing or wrong-type values raise
    immediately, not later when the broken code path is hit.
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # load from .env if it exists on disk
        env_file_encoding="utf-8",
        case_sensitive=False,      # API_PORT and api_port are the same
        extra="ignore",            # unknown env vars are silently ignored
    )

    # ── Application ───────────────────────────────────────────────────────────
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_format: str = "pretty"     # "pretty" = coloured console, "json" = production

    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: str = "ollama"   # "ollama" | "anthropic"

    # Ollama — free, runs locally
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Anthropic — paid, cloud (optional)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    # ── Agent behaviour ───────────────────────────────────────────────────────
    agent_max_tokens: int = 1024
    agent_temperature: float = 0.1
    agent_memory_window: int = 20
    agent_prompt_version: str = "v1"
    agent_retrieval_top_k: int = 5
    agent_low_confidence_threshold: float = 0.4

    # ── Vector store ──────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "insurance_kb"

    # ── Mock CRM ──────────────────────────────────────────────────────────────
    mock_crm_url: str = "http://localhost:8002"
    mock_crm_api_key: str = "mock-crm-secret-key"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql://resolveai:changeme-in-prod@localhost:5432/resolveai"
    )

    # ── Week 2+ (blank defaults so startup never fails) ───────────────────────
    zendesk_subdomain: str = ""
    zendesk_email: str = ""
    zendesk_api_token: str = ""
    slack_webhook_url: str = ""
    google_credentials_file: str = "credentials/google_credentials.json"
    google_calendar_id: str = "primary"

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def llm_model(self) -> str:
        """Active model name — works regardless of which provider is selected."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.ollama_model

    @property
    def is_production(self) -> bool:
        """True when running in production — used to enforce stricter behaviour."""
        return self.environment == "production"


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiated once when this module is first imported.
# Every `from src.config import settings` gets this exact object.
settings = Settings()