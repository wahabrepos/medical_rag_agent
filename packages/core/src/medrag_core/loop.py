"""The Self-MedRAG loop: retrieve, generate, verify, refine (Algorithm 1).

A framework-free reference implementation of the research work's `Trainer.run()`.
Retrieval, generation and verification are injected, so the same rules drive the
LangGraph agent and can be tested without models or network calls.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from medrag_core.policy import Decision, LoopSettings, decide, refine_query
from medrag_core.verification import Verification

logger = logging.getLogger(__name__)

ERROR_ANSWER = "Error during generation"


@dataclass(frozen=True)
class Generation:
    answer: str
    rationale: list[str]
    confidence: float
    citations: list[str]


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    query: str
    context: list[str]
    answer: str
    rationale: list[str]
    confidence: float
    support_score: float
    citations: list[str]


class StopReason(StrEnum):
    ACCEPTED = "accepted"
    STALLED = "stalled"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class LoopResult:
    answer: str
    rationale: list[str]
    iterations: int
    support_score: float
    stop_reason: StopReason
    history: list[IterationRecord]

    @property
    def confidence(self) -> float | None:
        """Confidence of the last iteration, which is what the research work reported.

        This can differ from the chosen answer's confidence when the best-of-history
        answer is returned.
        """
        return self.history[-1].confidence if self.history else None

    @property
    def citations(self) -> list[str] | None:
        """Citations of the last iteration (same caveat as `confidence`)."""
        return self.history[-1].citations if self.history else None


Retriever = Callable[[str], list[str]]
Generator = Callable[[str, list[str], list[IterationRecord]], Generation]
Verifier = Callable[[list[str], list[str]], Verification]


def run_self_medrag(
    query: str,
    *,
    retrieve: Retriever,
    generate: Generator,
    verify: Verifier,
    settings: LoopSettings | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> LoopResult:
    """Answer one question with iterative self-verification.

    Stops when the support score reaches the threshold, when it stops improving,
    after `max_iterations`, on timeout, or on the first error.
    """
    settings = settings or LoopSettings()
    start = clock()
    iteration = 0
    history: list[IterationRecord] = []
    current_query = query.strip()

    answer, rationale, support = "", [], 0.0
    stop_reason = StopReason.MAX_ITERATIONS

    while iteration < settings.max_iterations:
        if clock() - start > settings.max_time_seconds:
            logger.warning("Timeout after %d iterations", iteration)
            stop_reason = StopReason.TIMEOUT
            break
        try:
            context = retrieve(current_query)
            generation = generate(current_query, context, list(history))
            verification = verify(generation.rationale, context)
            record = IterationRecord(
                iteration=iteration + 1,
                query=current_query,
                context=context,
                answer=generation.answer,
                rationale=generation.rationale,
                confidence=generation.confidence,
                support_score=verification.support_score,
                citations=generation.citations,
            )
            history.append(record)

            decision = decide(history, settings)
            if decision is not Decision.CONTINUE:
                answer, rationale, support = record.answer, record.rationale, record.support_score
                stop_reason = (
                    StopReason.ACCEPTED if decision is Decision.ACCEPT else StopReason.STALLED
                )
                iteration += 1
                break

            # The research work refines from the original, unstripped query.
            refined = refine_query(query, verification.unsupported, iteration + 1, settings)
            if refined is not None:
                current_query = refined
            iteration += 1
        except Exception as exc:
            logger.exception("Error in iteration %d", iteration + 1)
            if iteration > 0:
                last = history[-1]
                answer, rationale, support = last.answer, last.rationale, last.support_score
            else:
                answer, rationale, support = ERROR_ANSWER, [str(exc)], 0.0
            stop_reason = StopReason.ERROR
            iteration += 1
            break

    if iteration == settings.max_iterations and not answer:
        # Iterations exhausted without an accepted answer: return the best-supported one.
        if history:
            best = max(history, key=lambda r: r.support_score)
            answer, rationale, support = best.answer, best.rationale, best.support_score
        else:
            answer, rationale, support = (
                "Unable to generate answer",
                ["No valid iterations completed"],
                0.0,
            )
        stop_reason = StopReason.MAX_ITERATIONS

    return LoopResult(
        answer=answer,
        rationale=rationale,
        iterations=iteration,
        support_score=support,
        stop_reason=stop_reason,
        history=history,
    )
