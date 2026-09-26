"""How well the retrieved literature supports an answer, for users.

Uses the NLI probabilities the verifier already computes. Every rationale
statement gets its best entailment and contradiction scores over the passages,
and the answer gets one status:

- supported: at least `threshold` of the statements are entailed by a passage
- partially_supported: some statements are entailed
- not_supported: none are (the answer rests on the model's own knowledge)
- contradicted: a passage contradicts a statement more than any passage supports it
- no_evidence: nothing was retrieved or there is no rationale to check

For yes/no questions the product may replace an unsupported answer with
"uncertain" (see `present_answer`); benchmarks keep the model's answer.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

EVIDENCE_THRESHOLD = 0.7
# Column order of the NLI model's probabilities (cross-encoder/nli-deberta-v3-base).
CONTRADICTION, ENTAILMENT = 0, 1

ProbabilityFn = Callable[[list[tuple[str, str]]], Sequence[Sequence[float]]]
"""(passage, statement) pairs -> [contradiction, entailment, neutral] per pair."""


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    NO_EVIDENCE = "no_evidence"


class AnswerFormat(StrEnum):
    FREE = "free"
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"


UNCERTAIN = "uncertain"

_PASSAGE_REF = r"(?:passages?|\[passage)\s*\d+\]?(?:\s*(?:,|and|&)\s*\d+\]?)*"
_VERBS = (
    r"(?:states?|shows?|reports?|notes?|indicates?|describes?|details?|mentions?|suggests?|"
    r"found|finds|demonstrates?|confirms?|highlights?|explains?|discusses?|provides?|"
    r"presents?|supports?|says|observes?|concludes?)"
)
_LEAD_IN = re.compile(
    rf"^\s*(?:(?:according to|as (?:stated|shown|reported) in|based on|from|in)\s+{_PASSAGE_REF}"
    rf"\s*,?\s*|{_PASSAGE_REF}\s+{_VERBS}\s+(?:that\s+)?)",
    re.IGNORECASE,
)
_INLINE_REF = re.compile(rf"\s*[\(\[]\s*{_PASSAGE_REF}\s*[\)\]]", re.IGNORECASE)


def normalize_statement(statement: str) -> str:
    """The factual claim of a rationale statement, without references to passages.

    Models often write "Passage 2 states that X"; NLI then judges a claim about a
    document rather than X, which no passage entails and other articles can seem to
    contradict. The claim X is what should be checked against the literature.
    """
    text = _INLINE_REF.sub("", _LEAD_IN.sub("", statement)).strip()
    return text[:1].upper() + text[1:] if text else statement.strip()


@dataclass(frozen=True)
class StatementEvidence:
    statement: str
    support: float  # best entailment probability over the passages
    supporting_passage: int | None  # index of that passage if support >= threshold
    contradiction: float  # best contradiction probability
    contradicting_passage: int | None  # index if contradiction >= threshold

    @property
    def supported(self) -> bool:
        return self.supporting_passage is not None

    @property
    def contradicted(self) -> bool:
        return self.contradicting_passage is not None and self.contradiction > self.support


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    supported_fraction: float
    statements: list[StatementEvidence]


def assess_evidence(
    rationale: Sequence[str],
    passages: Sequence[str],
    probabilities: ProbabilityFn,
    *,
    threshold: float = EVIDENCE_THRESHOLD,
    normalize: bool = True,
) -> EvidenceAssessment:
    """Assess each statement; with `normalize`, passage references are removed first
    (the original statement text is kept in the result)."""
    statements = [s for s in rationale if s.strip()]
    if not statements or not passages:
        return EvidenceAssessment(EvidenceStatus.NO_EVIDENCE, 0.0, [])

    claims = [normalize_statement(s) if normalize else s for s in statements]
    pairs = [(p, c) for c in claims for p in passages]
    rows = probabilities(pairs)
    n = len(passages)
    result: list[StatementEvidence] = []
    for i, statement in enumerate(statements):
        block = rows[i * n : (i + 1) * n]
        entail = [float(r[ENTAILMENT]) for r in block]
        contra = [float(r[CONTRADICTION]) for r in block]
        best_e = max(range(n), key=lambda j: entail[j])
        best_c = max(range(n), key=lambda j: contra[j])
        result.append(
            StatementEvidence(
                statement=statement,
                support=entail[best_e],
                supporting_passage=best_e if entail[best_e] >= threshold else None,
                contradiction=contra[best_c],
                contradicting_passage=best_c if contra[best_c] >= threshold else None,
            )
        )

    fraction = sum(s.supported for s in result) / len(result)
    if any(s.contradicted for s in result):
        status = EvidenceStatus.CONTRADICTED
    elif fraction >= threshold:
        status = EvidenceStatus.SUPPORTED
    elif fraction > 0:
        status = EvidenceStatus.PARTIALLY_SUPPORTED
    else:
        status = EvidenceStatus.NOT_SUPPORTED
    return EvidenceAssessment(status, fraction, result)


def present_answer(
    answer: str,
    status: EvidenceStatus,
    answer_format: AnswerFormat,
    *,
    allow_uncertain: bool = True,
) -> str:
    """The answer shown to users.

    For yes/no questions without supporting literature the model's yes/no is not
    evidence of anything (on PubMedQA without the source abstract the model answered
    "no" 73 times out of 75), so the product says "uncertain" instead.
    """
    unsupported = status in (EvidenceStatus.NOT_SUPPORTED, EvidenceStatus.NO_EVIDENCE)
    if allow_uncertain and answer_format is AnswerFormat.YES_NO and unsupported:
        return UNCERTAIN
    return answer
