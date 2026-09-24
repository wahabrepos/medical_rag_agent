"""Decisions of the Self-MedRAG loop: when to stop and how to refine the query."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class RefinementStrategy(StrEnum):
    STRUCTURED = "structured"
    DECOMPOSITION = "decomposition"
    CONCATENATION = "concatenation"


class Decision(StrEnum):
    ACCEPT = "accept"  # support score reached the threshold
    STALLED = "stalled"  # improvement over the previous iteration was too small
    CONTINUE = "continue"  # refine the query and try again


@dataclass(frozen=True)
class LoopSettings:
    """Research-work parity defaults (config_selfmedrag_mistral.yaml)."""

    max_iterations: int = 3
    early_stopping: bool = True
    min_improvement: float = 0.05
    max_time_seconds: float = 600.0
    rationale_score_threshold: float = 0.7
    refinement_strategy: RefinementStrategy = RefinementStrategy.STRUCTURED
    max_unsupported_in_query: int = 3


class Scored(Protocol):
    @property
    def support_score(self) -> float: ...


def decide(history: Sequence[Scored], settings: LoopSettings) -> Decision:
    """Decision after the latest iteration (the last entry of `history`)."""
    score = history[-1].support_score
    if score >= settings.rationale_score_threshold:
        return Decision.ACCEPT
    if (
        settings.early_stopping
        and len(history) > 1
        and score - history[-2].support_score < settings.min_improvement
    ):
        return Decision.STALLED
    return Decision.CONTINUE


def refine_query(
    original_query: str,
    unsupported: Sequence[str],
    iteration: int,
    settings: LoopSettings,
) -> str | None:
    """Next query built from the unsupported statements, or None to keep the current one.

    `iteration` is the 1-based number of the iteration that just finished.
    """
    if not unsupported:
        return None
    text = " ".join(unsupported[: settings.max_unsupported_in_query])
    match settings.refinement_strategy:
        case RefinementStrategy.STRUCTURED:
            return f"{original_query} Find specific evidence for: {text}"
        case RefinementStrategy.DECOMPOSITION:
            return f"Sub-question {iteration}: What evidence supports that {text}?"
        case RefinementStrategy.CONCATENATION:
            return f"{original_query} {text}"
