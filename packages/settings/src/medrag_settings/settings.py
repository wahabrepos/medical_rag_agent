"""Runtime configuration loaded from environment variables and `.env`.

Secrets are typed as `SecretStr` so they never appear in logs, reprs or
tracebacks. Call `.get_secret_value()` only at the point of use.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnv = AppEnv.LOCAL
    log_level: str = "INFO"

    # LLM providers (primary Mistral, fallback Groq), as LiteLLM model names.
    mistral_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_primary_model: str = "mistral/mistral-small-latest"
    llm_fallback_model: str = "groq/openai/gpt-oss-120b"

    # Hugging Face token for model downloads.
    hf_token: SecretStr | None = None

    # Storage.
    database_url: SecretStr = Field(
        default=SecretStr("postgresql+psycopg://medrag:medrag@localhost:5432/medrag")
    )
    redis_url: str = "redis://localhost:6379/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
