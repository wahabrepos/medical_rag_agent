from types import SimpleNamespace
from typing import Any

import pytest

from medrag_agent.errors import QuotaExhaustedError
from medrag_agent.llm import (
    GeneratorConfig,
    LlmGenerator,
    RateLimiter,
    config_for_model,
)
from medrag_core.prompts import build_messages

JSON_ANSWER = '{"answer": "B", "rationale": ["one", "two"], "confidence": 0.8}'


class FakeCompletion:
    def __init__(self, text: str = JSON_ANSWER, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            model="groq/openai/gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text), finish_reason=self.finish_reason
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


class NoWait(RateLimiter):
    def __init__(self) -> None:
        super().__init__(1000, 10**9)


def test_sends_research_work_call_and_parses_answer() -> None:
    fake = FakeCompletion()
    gen = LlmGenerator(GeneratorConfig(), api_key="k", completion=fake, limiter=NoWait())

    result = gen("Q?", ["passage"], [], binary_answer=True)

    expected = build_messages("Q?", ["passage"], [], binary_answer=True)
    call = fake.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": expected.system},
        {"role": "user", "content": expected.user},
    ]
    assert call["max_tokens"] == 400
    assert "temperature" not in call
    assert "reasoning_effort" not in call
    assert call["api_key"] == "k"
    assert (result.answer, result.rationale, result.confidence) == ("B", ["one", "two"], 0.8)
    assert gen.calls == 1


def test_temperature_is_sent_only_when_configured() -> None:
    fake = FakeCompletion()
    LlmGenerator(GeneratorConfig(temperature=0.0), completion=fake, limiter=NoWait())("Q", [], [])
    assert fake.calls[0]["temperature"] == 0.0


def test_unparseable_output_uses_research_work_fallback() -> None:
    gen = LlmGenerator(GeneratorConfig(), completion=FakeCompletion("Answer: C"), limiter=NoWait())
    assert gen("Q", [], []).answer == "C"


def test_rate_limiter_waits_for_the_window() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, 1000, clock=lambda: now[0], sleep=sleep)
    limiter.acquire(400)
    limiter.acquire(400)
    limiter.acquire(400)  # third request exceeds 2 per minute and the token budget

    assert slept
    assert now[0] >= 60.0


class RateLimitError(Exception):
    """Shaped like litellm.RateLimitError."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


GROQ_DAILY = (
    "GroqException - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day "
    "(TPD): Limit 200000, Used 198858, Requested 1779. Please try again in 4m35s."
)


@pytest.mark.parametrize("message", [GROQ_DAILY, "Too many requests per minute"])
def test_rate_limits_after_retries_are_reported_as_quota(message: str) -> None:
    def failing(**kwargs: Any) -> Any:
        raise RateLimitError(message)

    gen = LlmGenerator(GeneratorConfig(), completion=failing, limiter=NoWait())
    with pytest.raises(QuotaExhaustedError):
        gen("Q", [], [])


def test_other_errors_are_not_quota() -> None:
    def failing(**kwargs: Any) -> Any:
        raise ValueError("bad request")

    gen = LlmGenerator(GeneratorConfig(), completion=failing, limiter=NoWait())
    with pytest.raises(ValueError, match="bad request"):
        gen("Q", [], [])


def test_reasoning_models_get_headroom_and_low_effort() -> None:
    fake = FakeCompletion()
    config = config_for_model("groq/openai/gpt-oss-120b", tokens_per_minute=5000)
    LlmGenerator(config, completion=fake, limiter=NoWait())("Q", [], [])

    assert config.tokens_per_minute == 5000
    assert fake.calls[0]["max_tokens"] == 2048
    assert fake.calls[0]["reasoning_effort"] == "low"


def test_other_models_keep_research_work_settings() -> None:
    config = config_for_model("mistral/mistral-small-2603")
    assert (config.max_tokens, config.reasoning_effort) == (400, None)
