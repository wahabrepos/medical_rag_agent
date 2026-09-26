"""How well the retrieved literature supports an answer, for users.

Uses the NLI probabilities the verifier already computes. Every rationale
statement gets its best entailment and contradiction scores over the passages,
and the answer gets one status:

- supported: at least SUPPORTED_SHARE of the statements are entailed by a passage
- partially_supported: some statements are entailed
- not_supported: none are (the answer rests on the model's own knowledge)
- contradicted: a passage contradicts a statement more than any passage supports it
- no_evidence: nothing was retrieved or there is no rationale to check

What users see depends on the answer policy (see `present_answer`): the product
shows an answer only when the literature supports all of it; benchmarks keep the
model's answer.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

EVIDENCE_THRESHOLD = 0.7  # support probability of one statement-passage pair
SUPPORTED_SHARE = 0.7  # share of supported statements for "supported"
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
INSUFFICIENT_EVIDENCE = "insufficient evidence"


class AnswerPolicy(StrEnum):
    # Show an answer only when every checked statement is supported by a passage.
    EVIDENCE_GATED = "evidence_gated"
    # v3a behaviour: unsupported yes/no answers become "uncertain", others are shown.
    UNCERTAIN_YES_NO = "uncertain_yes_no"
    # Always the model's answer (benchmarks).
    SHOW_ALL = "show_all"


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
class QuotedClaim:
    """A claim with the passage (1-based, as numbered in the prompt) and the verbatim
    sentence the model says supports it."""

    claim: str
    passage: int | None
    quote: str


@dataclass(frozen=True)
class StatementEvidence:
    statement: str
    support: float  # best entailment probability over the passages
    supporting_passage: int | None  # index of that passage if support >= threshold
    contradiction: float  # best contradiction probability
    contradicting_passage: int | None  # index if contradiction >= threshold
    quote: str | None = None  # verbatim quote from the supporting passage (quoted mode)

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
    (the original statement text is kept in the result). `threshold` is the pair-level
    probability that counts as support or contradiction; it depends on the verifier."""
    statements = base_statements(rationale)
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

    return EvidenceAssessment(_status(result), _fraction(result), result)


def _fraction(statements: Sequence[StatementEvidence]) -> float:
    return sum(s.supported for s in statements) / len(statements)


def _status(statements: Sequence[StatementEvidence]) -> EvidenceStatus:
    fraction = _fraction(statements)
    if any(s.contradicted for s in statements):
        return EvidenceStatus.CONTRADICTED
    if fraction >= SUPPORTED_SHARE:
        return EvidenceStatus.SUPPORTED
    if fraction > 0:
        return EvidenceStatus.PARTIALLY_SUPPORTED
    return EvidenceStatus.NOT_SUPPORTED


MIN_QUOTE_CHARS = 20
_QUOTE_CHARS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def _squash(text: str) -> str:
    return " ".join(text.translate(_QUOTE_CHARS).lower().split())


# Sentences stating what a study set out to do are not findings: "the aim was to
# examine whether elderly patients have different needs" supports nothing about
# their needs, yet verifiers score such pairs as supported.
_AIM = re.compile(
    r"\b(?:aims?|objectives?|purpose|goals?)\b[^.]{0,40}\b(?:is|was|were|are)\s+to\b"
    r"|\baimed\s+to\b"
    r"|\bwe\s+(?:hypothesi[sz]ed|sought|aimed|set\s+out)\b"
    r"|\b(?:to\s+(?:determine|assess|evaluate|investigate|examine|compare|test|explore))"
    r"\s+(?:whether|if)\b",
    re.IGNORECASE,
)


def states_aim(quote: str) -> bool:
    """Whether a quote describes a study's aim or hypothesis rather than a result."""
    return _AIM.search(quote) is not None


def quote_in_passage(quote: str, passage: str) -> bool:
    """Whether `quote` is copied from `passage`, ignoring case, whitespace, typographic
    quotes and dashes; "..." may join parts that appear in order."""
    target = _squash(passage)
    parts = [_squash(p).strip(" .,;:\"'") for p in re.split(r"\.\.\.|\u2026|\[\.\.\.\]", quote)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return False
    position = 0
    for part in parts:
        found = target.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def assess_quoted_claims(
    rationale: Sequence[str],
    quotes: Sequence[QuotedClaim],
    passages: Sequence[str],
    probabilities: ProbabilityFn,
    *,
    threshold: float = EVIDENCE_THRESHOLD,
    normalize: bool = True,
) -> EvidenceAssessment:
    """Like `assess_evidence`, but a statement counts as supported only when the model
    quoted a sentence that really is in a retrieved passage, the quote reports a
    result (not a study aim or hypothesis), and the verifier finds that passage
    supports the statement. Statements without a valid quote are unsupported,
    whatever the verifier says about other passages."""
    base = assess_evidence(
        rationale, passages, probabilities, threshold=threshold, normalize=normalize
    )
    if base.status is EvidenceStatus.NO_EVIDENCE:
        return base
    by_claim: dict[str, list[QuotedClaim]] = {}
    for q in quotes:
        by_claim.setdefault(_squash(normalize_statement(q.claim)), []).append(q)
    claims = [normalize_statement(s) if normalize else s for s in base_statements(rationale)]
    candidates: list[tuple[int, int, str]] = []  # (statement, passage index, quote)
    for i, claim in enumerate(claims):
        for q in by_claim.get(_squash(normalize_statement(claim)), []):
            cited = q.passage - 1 if q.passage and 0 < q.passage <= len(passages) else None
            order = [cited] if cited is not None else []
            order += [j for j in range(len(passages)) if j != cited]
            if states_aim(q.quote):
                continue
            for j in order:
                if quote_in_passage(q.quote, passages[j]):
                    candidates.append((i, j, q.quote))
                    break
    rows = probabilities([(passages[j], claims[i]) for i, j, _ in candidates]) if candidates else []
    best: dict[int, tuple[float, int, str]] = {}
    for (i, j, quote), row in zip(candidates, rows, strict=True):
        support = float(row[ENTAILMENT])
        if i not in best or support > best[i][0]:
            best[i] = (support, j, quote)
    result = []
    for i, s in enumerate(base.statements):
        support, j, quote = best.get(i, (0.0, -1, ""))
        ok = support >= threshold
        result.append(
            StatementEvidence(
                statement=s.statement,
                support=support,
                supporting_passage=j if ok else None,
                contradiction=s.contradiction,
                contradicting_passage=s.contradicting_passage,
                quote=quote if ok else None,
            )
        )
    return EvidenceAssessment(_status(result), _fraction(result), result)


def base_statements(rationale: Sequence[str]) -> list[str]:
    return [s for s in rationale if s.strip()]


def present_answer(
    answer: str,
    assessment: EvidenceAssessment,
    answer_format: AnswerFormat,
    *,
    policy: AnswerPolicy = AnswerPolicy.EVIDENCE_GATED,
) -> str:
    """The answer shown to users.

    Gated (the product default): only an answer whose every checked statement is
    supported is shown; anything else is "insufficient evidence". On the full
    evaluation run such answers were 94% correct on PubMedQA, against 72% for the
    rest, so an unsupported answer is not evidence of anything.

    For yes/no questions without supporting literature the model's yes/no is not
    evidence of anything either (on PubMedQA without the source abstract the model
    answered "no" 73 times out of 75); the UNCERTAIN_YES_NO policy says "uncertain".
    """
    match policy:
        case AnswerPolicy.SHOW_ALL:
            return answer
        case AnswerPolicy.EVIDENCE_GATED:
            grounded = assessment.statements and all(s.supported for s in assessment.statements)
            return answer if grounded else INSUFFICIENT_EVIDENCE
        case AnswerPolicy.UNCERTAIN_YES_NO:
            unsupported = assessment.status in (
                EvidenceStatus.NOT_SUPPORTED,
                EvidenceStatus.NO_EVIDENCE,
            )
            if answer_format is AnswerFormat.YES_NO and unsupported:
                return UNCERTAIN
            return answer
