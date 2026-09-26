import pytest

from medrag_core.evidence import (
    INSUFFICIENT_EVIDENCE,
    UNCERTAIN,
    AnswerFormat,
    AnswerPolicy,
    EvidenceAssessment,
    EvidenceStatus,
    StatementEvidence,
    assess_evidence,
    present_answer,
)


def table(rows: dict[tuple[str, str], tuple[float, float, float]]):  # type: ignore[no-untyped-def]
    def probabilities(pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        return [rows.get(p, (0.0, 0.0, 1.0)) for p in pairs]

    return probabilities


def test_supported_when_most_statements_are_entailed() -> None:
    probs = table({("p1", "S1"): (0.0, 0.9, 0.1), ("p2", "S2"): (0.0, 0.8, 0.2)})
    result = assess_evidence(["S1", "S2", "S3"], ["p1", "p2"], probs)

    assert result.status is EvidenceStatus.PARTIALLY_SUPPORTED  # 2/3 < 0.7
    assert [s.supporting_passage for s in result.statements] == [0, 1, None]
    assert assess_evidence(["S1", "S2"], ["p1", "p2"], probs).status is EvidenceStatus.SUPPORTED


def test_not_supported_when_nothing_is_entailed() -> None:
    result = assess_evidence(["S1"], ["p1"], table({}))
    assert result.status is EvidenceStatus.NOT_SUPPORTED
    assert result.supported_fraction == 0.0


def test_contradiction_wins_over_support() -> None:
    probs = table({("p1", "S1"): (0.0, 0.9, 0.1), ("p2", "S2"): (0.85, 0.05, 0.1)})
    result = assess_evidence(["S1", "S2"], ["p1", "p2"], probs)

    assert result.status is EvidenceStatus.CONTRADICTED
    assert result.statements[1].contradicting_passage == 1


def test_contradiction_weaker_than_support_is_ignored() -> None:
    probs = table({("p1", "S1"): (0.75, 0.0, 0.25), ("p2", "S1"): (0.0, 0.8, 0.2)})
    assert assess_evidence(["S1"], ["p1", "p2"], probs).status is EvidenceStatus.SUPPORTED


@pytest.mark.parametrize(("rationale", "passages"), [([], ["p"]), (["s"], []), ([" "], ["p"])])
def test_no_evidence(rationale: list[str], passages: list[str]) -> None:
    assert assess_evidence(rationale, passages, table({})).status is EvidenceStatus.NO_EVIDENCE


def assessment(*supported: bool, status: EvidenceStatus | None = None) -> EvidenceAssessment:
    statements = [
        StatementEvidence(f"s{i}", 0.9 if ok else 0.1, 0 if ok else None, 0.0, None)
        for i, ok in enumerate(supported)
    ]
    fraction = sum(supported) / len(supported) if supported else 0.0
    if status is None:
        status = (
            EvidenceStatus.NO_EVIDENCE
            if not supported
            else EvidenceStatus.SUPPORTED
            if fraction >= 0.7
            else EvidenceStatus.PARTIALLY_SUPPORTED
            if fraction > 0
            else EvidenceStatus.NOT_SUPPORTED
        )
    return EvidenceAssessment(status, fraction, statements)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (assessment(True, True), "B"),
        (assessment(True, True, True, False), INSUFFICIENT_EVIDENCE),  # "supported" at 75%
        (assessment(True, False), INSUFFICIENT_EVIDENCE),
        (assessment(False), INSUFFICIENT_EVIDENCE),
        (assessment(), INSUFFICIENT_EVIDENCE),
    ],
)
def test_gated_answers_need_every_statement_supported(
    evidence: EvidenceAssessment, expected: str
) -> None:
    for fmt in AnswerFormat:
        assert present_answer("B", evidence, fmt) == expected


@pytest.mark.parametrize(
    ("evidence", "fmt", "expected"),
    [
        (assessment(False), AnswerFormat.YES_NO, UNCERTAIN),
        (assessment(), AnswerFormat.YES_NO, UNCERTAIN),
        (assessment(True, False), AnswerFormat.YES_NO, "no"),
        (assessment(False), AnswerFormat.MULTIPLE_CHOICE, "no"),
        (assessment(False), AnswerFormat.FREE, "no"),
    ],
)
def test_uncertain_yes_no_policy(
    evidence: EvidenceAssessment, fmt: AnswerFormat, expected: str
) -> None:
    assert present_answer("no", evidence, fmt, policy=AnswerPolicy.UNCERTAIN_YES_NO) == expected


def test_show_all_keeps_the_model_answer() -> None:
    assert (
        present_answer("no", assessment(False), AnswerFormat.YES_NO, policy=AnswerPolicy.SHOW_ALL)
        == "no"
    )


@pytest.mark.parametrize(
    ("statement", "claim"),
    [
        (
            "Passage 1 states that PCD occurs in lace plant leaves.",
            "PCD occurs in lace plant leaves.",
        ),
        ("Passage 2 details a study of mitochondria.", "A study of mitochondria."),
        ("Passages 1 and 3 show that aspirin helps.", "Aspirin helps."),
        ("According to passage 4, metformin is first line.", "Metformin is first line."),
        ("Metformin is first line (Passage 2).", "Metformin is first line."),
        ("[Passage 1] reports that X is common.", "X is common."),
        ("Metformin is first line.", "Metformin is first line."),
        ("The passage count was 5.", "The passage count was 5."),
    ],
)
def test_normalize_statement(statement: str, claim: str) -> None:
    from medrag_core.evidence import normalize_statement

    assert normalize_statement(statement) == claim


def test_assessment_checks_the_claim_but_reports_the_original() -> None:
    probs = table({("p1", "Aspirin helps."): (0.0, 0.9, 0.1)})
    result = assess_evidence(["Passage 1 shows that aspirin helps."], ["p1"], probs)

    assert result.status is EvidenceStatus.SUPPORTED
    assert result.statements[0].statement == "Passage 1 shows that aspirin helps."
    raw = assess_evidence(["Passage 1 shows that aspirin helps."], ["p1"], probs, normalize=False)
    assert raw.status is EvidenceStatus.NOT_SUPPORTED
