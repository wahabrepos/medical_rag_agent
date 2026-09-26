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

    # LLM (LiteLLM model names). Groq gpt-oss-120b while Mistral-small is not on the
    # free plan; the research work used mistral/mistral-small-latest.
    mistral_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_primary_model: str = "groq/openai/gpt-oss-120b"
    llm_fallback_model: str = "mistral/mistral-small-2603"
    # Client-side throttle; defaults match Groq's free tier for gpt-oss-120b.
    llm_requests_per_minute: int = 30
    llm_tokens_per_minute: int = 8_000

    # Hugging Face token for model downloads.
    hf_token: SecretStr | None = None

    # Inference service (embeddings + NLI on CPU). None = ONNX Runtime default.
    inference_threads: int | None = None
    inference_device: str = "cpu"  # "cuda" needs onnxruntime-gpu

    # Public API (apps/api).
    api_keys: SecretStr | None = None  # comma-separated bearer keys; none = open in APP_ENV=local
    api_requests_per_minute: int = 20  # per key
    allow_uncertain: bool = True  # yes/no answers without supporting literature -> "uncertain"
    nli_url: str | None = None  # remote inference service for NLI; unset = in-process
    llm_budget: float = 3.0  # total LLM spend cap shared with evaluations (stops at 90%)
    llm_spend_ledger: str = "data/llm_spend.json"

    # Storage.
    database_url: SecretStr = Field(
        default=SecretStr("postgresql+psycopg://medrag:medrag@localhost:5432/medrag")
    )
    redis_url: str = "redis://localhost:6379/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
