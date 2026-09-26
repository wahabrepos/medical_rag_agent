"""Generator: one chat-completion call per loop iteration, through LiteLLM.

Reproduces the research work's API call: the same system and user messages,
`max_tokens=400`, no temperature (the provider's default applies) and a single
call whose text is parsed with the research-work JSON parser.

Reasoning models (gpt-oss) count hidden reasoning against `max_tokens`; with the
research work's 400 they often return an empty answer. For them the call uses
`reasoning_effort="low"` and 2,048 tokens of headroom (answers stay about the same
size). See `config_for_model`.

A client-side limiter keeps requests within the provider's per-minute limits,
and a daily-quota error is raised as `QuotaExhaustedError` so evaluation runs can stop
cleanly and resume later.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from medrag_agent.errors import QuotaExhaustedError
from medrag_core.loop import Generation
from medrag_core.parsing import ParsedGeneration
from medrag_core.prompts import ChatMessages, HistoryEntry, build_messages

logger = logging.getLogger(__name__)

RESEARCH_WORK_MAX_TOKENS = 400
REASONING_MAX_TOKENS = 2048
CHARS_PER_TOKEN = 4  # rough estimate used only for throttling


@dataclass(frozen=True)
class GeneratorConfig:
    model: str = "groq/openai/gpt-oss-120b"
    max_tokens: int = RESEARCH_WORK_MAX_TOKENS
    temperature: float | None = None  # None: not sent, as in the research work
    reasoning_effort: str | None = None  # sent only when set (reasoning models)
    expected_completion_tokens: int = 400  # used only to estimate throttling
    lenient_json: bool = False  # read fields from invalid JSON (not in the research work)
    answer_claim: bool = False  # ask for a one-sentence answer claim (answer check)
    requests_per_minute: int = 30
    tokens_per_minute: int = 8_000
    num_retries: int = 6
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class LlmResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


class RateLimiter:
    """Sliding one-minute window over requests and (estimated) tokens."""

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= 60.0:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    self._events.append((now, tokens))
                    return
                wait = 60.0 - (now - self._events[0][0]) + 0.05
            self._sleep(max(wait, 0.05))


CompletionFn = Callable[..., Any]


@dataclass
class LlmGenerator:
    """Callable generator for the Self-MedRAG loop (see medrag_core.loop.Generator)."""

    config: GeneratorConfig
    api_key: str | None = None
    completion: CompletionFn | None = None
    limiter: RateLimiter | None = None
    calls: int = field(default=0, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    # Raw text of every call, for diagnosing parser fallbacks.
    raw_outputs: list[str] = field(default_factory=list, init=False)
    last_response: LlmResponse | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.completion is None:
            import litellm

            litellm.suppress_debug_info = True
            self.completion = litellm.completion
        self._completion: CompletionFn = self.completion
        self._limiter = self.limiter or RateLimiter(
            self.config.requests_per_minute, self.config.tokens_per_minute
        )

    def complete(self, messages: ChatMessages) -> LlmResponse:
        estimate = (
            len(messages.system) + len(messages.user)
        ) // CHARS_PER_TOKEN + self.config.max_tokens
        self._limiter.acquire(estimate)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": messages.system},
                {"role": "user", "content": messages.user},
            ],
            "max_tokens": self.config.max_tokens,
            "num_retries": self.config.num_retries,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            response = self._completion(**kwargs)
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise QuotaExhaustedError(str(exc)) from exc
            raise
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        result = LlmResponse(
            text=choice.message.content or "",
            model=getattr(response, "model", self.config.model) or self.config.model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
        )
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.raw_outputs.append(result.text)
        if result.finish_reason == "length":
            logger.warning("generation hit max_tokens=%d", self.config.max_tokens)
        self.last_response = result
        return result

    def __call__(
        self,
        query: str,
        context: Sequence[str],
        history: Sequence[HistoryEntry],
        *,
        binary_answer: bool = False,
        multiple_choice: bool = False,
    ) -> Generation:
        messages = build_messages(
            query,
            context,
            history,
            binary_answer=binary_answer,
            multiple_choice=multiple_choice,
            answer_claim=self.config.answer_claim,
        )
        parsed = ParsedGeneration.from_text(
            self.complete(messages).text, lenient=self.config.lenient_json
        )
        return Generation(
            answer=parsed.answer,
            rationale=parsed.rationale,
            confidence=parsed.confidence,
            citations=parsed.citations,
            claim=parsed.claim if self.config.answer_claim else None,
        )


def is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model


def config_for_model(model: str, **overrides: Any) -> GeneratorConfig:
    """Research-work call settings, with reasoning headroom for reasoning models."""
    if is_reasoning_model(model):
        base = GeneratorConfig(
            model=model,
            max_tokens=REASONING_MAX_TOKENS,
            reasoning_effort="low",
            expected_completion_tokens=600,
        )
    else:
        base = GeneratorConfig(model=model)
    return replace(base, **overrides)


def is_rate_limit_error(exc: Exception) -> bool:
    """A 429 / rate-limit error. LiteLLM has already retried it, so the quota is used up.

    Covers per-minute and per-day limits alike (Groq reports its daily token limit
    with a short "try again in 4m" wait while the day's budget stays exhausted).
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ == "RateLimitError" or "rate limit" in str(exc).lower()
