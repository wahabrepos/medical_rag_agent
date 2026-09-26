"""Evidence assessment of the answer an agent run returned.

Shared by the API (what users see) and the evaluation runner (so evaluations report
exactly what the product would have shown).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from medrag_core.evidence import (
    EvidenceAssessment,
    assess_evidence,
    assess_quoted_claims,
    ground_in_question,
)


class Verifier(Protocol):
    support_threshold: float

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = ...
    ) -> npt.NDArray[np.float32] | Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class FinalAssessment:
    assessment: EvidenceAssessment
    passages: list[Any]  # passages of the iteration whose answer was returned
    has_claim: bool  # the first assessed statement is the answer claim


def assess_final(state: Any, verifier: Verifier, *, quoted: bool = False) -> FinalAssessment:
    """Assess the returned answer against the passages it was generated from.

    With an answer claim, the claim is assessed first, then the rationale. With
    `quoted`, a statement counts only when its verbatim quote is found in a passage
    (see medrag_core.evidence.assess_quoted_claims). Statements that restate the
    question are then marked as grounded in it (medrag_core.evidence.ground_in_question).
    """
    final = state.get("final_iteration")
    passages: list[Any] = list(state["iteration_passages"][final - 1]) if final else []
    texts = [p if isinstance(p, str) else p.text for p in passages]
    claims = state.get("iteration_claims", [])
    claim = claims[final - 1] if final and len(claims) >= final else None
    rationale = list(state.get("rationale", []))
    statements = [claim, *rationale] if claim else rationale

    def probabilities(pairs: list[tuple[str, str]]) -> Any:
        return verifier.probabilities(pairs)

    if quoted:
        evidence = state.get("iteration_evidence", [])
        quotes = evidence[final - 1] if final and len(evidence) >= final else ()
        assessment = assess_quoted_claims(
            statements, quotes, texts, probabilities, threshold=verifier.support_threshold
        )
    else:
        assessment = assess_evidence(
            statements, texts, probabilities, threshold=verifier.support_threshold
        )
    assessment = ground_in_question(
        assessment,
        state.get("question", ""),
        probabilities,
        threshold=verifier.support_threshold,
    )
    return FinalAssessment(assessment, passages, claim is not None)
