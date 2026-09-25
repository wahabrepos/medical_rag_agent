"""Errors that must stop a run instead of being recorded as a question's answer."""


class ProviderUnavailableError(RuntimeError):
    """The LLM provider cannot serve requests right now (quota, rate limit).

    The graph lets this propagate instead of turning it into an "Error during
    generation" answer, so evaluation runs stop and the question is asked again
    later.
    """


class QuotaExhaustedError(ProviderUnavailableError):
    """A rate or quota limit that LiteLLM's retries could not get past."""
