import pytest

from medrag_settings import AppEnv, Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "APP_ENV",
        "LOG_LEVEL",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "HF_TOKEN",
        "LLM_PRIMARY_MODEL",
        "LLM_FALLBACK_MODEL",
        "DATABASE_URL",
        "REDIS_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env_file() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_env is AppEnv.LOCAL
    assert settings.llm_primary_model == "groq/openai/gpt-oss-120b"
    assert settings.llm_fallback_model == "mistral/mistral-small-2603"
    assert settings.mistral_api_key is None


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_env is AppEnv.PROD
    assert settings.mistral_api_key is not None
    assert settings.mistral_api_key.get_secret_value() == "test-mistral-key"


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert "test-groq-key" not in repr(settings)
    assert "test-groq-key" not in str(settings.model_dump())
