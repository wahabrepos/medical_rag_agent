"""Self-reflection: check rationale statements against retrieved passages with NLI.

A statement is supported when its best entailment probability over all passages
reaches the threshold. The support score is the share of supported statements.
The NLI model itself is injected, so this module stays framework-free.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

VERIFICATION_THRESHOLD = 0.7

NliScorer = Callable[[list[tuple[str, str]]], list[float]]
"""Entailment probability for each (premise passage, hypothesis statement) pair."""


@dataclass(frozen=True)
class Verification:
    support_score: float
    unsupported: list[str]
    best_scores: list[float]


def verify_rationale(
    rationale: Sequence[str],
    passages: Sequence[str],
    nli_scorer: NliScorer,
    *,
    threshold: float = VERIFICATION_THRESHOLD,
) -> Verification:
    """Score every (passage, statement) pair in one batch and aggregate per statement.

    An empty rationale counts as fully supported; with no passages, every statement
    is unsupported (both as in the research work).
    """
    if not rationale:
        return Verification(support_score=1.0, unsupported=[], best_scores=[])
    if not passages:
        return Verification(support_score=0.0, unsupported=list(rationale), best_scores=[])

    pairs = [(passage, statement) for statement in rationale for passage in passages]
    probs = nli_scorer(pairs)

    n = len(passages)
    unsupported: list[str] = []
    best_scores: list[float] = []
    for i, statement in enumerate(rationale):
        row = probs[i * n : (i + 1) * n]
        best = max(row) if row else 0.0
        best_scores.append(best)
        if best < threshold:
            unsupported.append(statement)

    supported = len(rationale) - len(unsupported)
    return Verification(
        support_score=supported / len(rationale),
        unsupported=unsupported,
        best_scores=best_scores,
    )
