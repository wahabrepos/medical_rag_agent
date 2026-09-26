import pytest

from medrag_core.evidence import (
    INSUFFICIENT_EVIDENCE,
    UNCERTAIN,
    AnswerFormat,
    AnswerPolicy,
    EvidenceAssessment,
    EvidenceStatus,
    QuotedClaim,
    StatementEvidence,
    assess_evidence,
    assess_quoted_claims,
    ground_in_question,
    is_grounded,
    present_answer,
    quote_in_passage,
    states_aim,
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


EN_DASH = chr(0x2013)  # as PubMed writes ranges
PASSAGE = (
    f"Metformin lowered HbA1c by 1.1% (95% CI 0.9{EN_DASH}1.3) in adults with type 2 diabetes. "
    "No serious adverse events were reported."
)


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        ("Metformin lowered HbA1c by 1.1% (95% CI 0.9-1.3)", True),  # dash normalised
        ("metformin   LOWERED hba1c by 1.1%", True),  # case and spaces
        ("Metformin lowered HbA1c ... no serious adverse events were reported.", True),
        ("No serious adverse events ... Metformin lowered HbA1c", False),  # wrong order
        ("Metformin raised HbA1c by 1.1%", False),
        ("Metformin", False),  # too short to count as a quote
    ],
)
def test_quote_in_passage(quote: str, found: bool) -> None:
    assert quote_in_passage(quote, PASSAGE) is found


def quoted_table(support: float):  # type: ignore[no-untyped-def]
    return table({(PASSAGE, "Metformin lowers HbA1c."): (0.0, support, 1 - support)})


def test_quoted_claim_needs_a_real_quote_and_verifier_support() -> None:
    claim = "Metformin lowers HbA1c."
    real = QuotedClaim(claim, 2, "Metformin lowered HbA1c by 1.1%")
    invented = QuotedClaim(claim, 2, "Metformin lowered HbA1c in every trial")
    passages = ["Unrelated passage about asthma.", PASSAGE]

    ok = assess_quoted_claims([claim], [real], passages, quoted_table(0.9))
    (statement,) = ok.statements
    assert ok.status is EvidenceStatus.SUPPORTED
    assert (statement.supporting_passage, statement.quote) == (1, real.quote)

    # The verifier alone would support it, but the quote is not in any passage.
    fake = assess_quoted_claims([claim], [invented], passages, quoted_table(0.9))
    assert fake.status is EvidenceStatus.NOT_SUPPORTED
    assert assess_evidence([claim], passages, quoted_table(0.9)).status is EvidenceStatus.SUPPORTED

    # A real quote the verifier does not accept, or no quote at all, is unsupported.
    weak = assess_quoted_claims([claim], [real], passages, quoted_table(0.2))
    none = assess_quoted_claims([claim], [], passages, quoted_table(0.9))
    assert weak.status is none.status is EvidenceStatus.NOT_SUPPORTED


def test_quote_found_in_another_passage_than_cited() -> None:
    claim = "Metformin lowers HbA1c."
    quote = QuotedClaim(claim, 1, "Metformin lowered HbA1c by 1.1%")  # cites passage 1

    result = assess_quoted_claims([claim], [quote], ["Asthma.", PASSAGE], quoted_table(0.9))

    assert result.statements[0].supporting_passage == 1  # found in passage 2 (index 1)


@pytest.mark.parametrize(
    ("quote", "aim"),
    [
        ("The aim of the present study is to examine whether elderly patients differ.", True),
        ("The purpose of this study was to verify the efficacy of quilting sutures.", True),
        ("We hypothesized that enhanced migration may be crucial.", True),
        ("This study was designed to determine whether preclerkship exams predict failure.", True),
        ("Our objective was to compare two techniques.", True),
        ("Seroma occurred in 2% of patients with quilting sutures versus 20% without.", False),
        ("Treatment aims were met in most patients (82%).", False),
        ("Metformin lowered HbA1c by 1.1% compared with placebo.", False),
    ],
)
def test_states_aim(quote: str, aim: bool) -> None:
    assert states_aim(quote) is aim


def test_aim_quotes_do_not_count_as_evidence() -> None:
    passage = "The aim of this study was to determine whether metformin lowers HbA1c."
    claim = "Metformin lowers HbA1c."
    probs = table({(passage, claim): (0.0, 0.95, 0.05)})

    result = assess_quoted_claims([claim], [QuotedClaim(claim, 1, passage)], [passage], probs)

    assert result.status is EvidenceStatus.NOT_SUPPORTED


QUESTION = "A 63-year-old man has chest pain, hypotension and an irregular pulse. Next step?"


def test_restated_findings_are_grounded_in_the_question() -> None:
    finding = "The patient is hypotensive with an irregular pulse."
    diagnosis = "The patient has aortic dissection."
    fact = "Unstable tachyarrhythmias need synchronized cardioversion."
    probs = table(
        {
            (QUESTION, finding): (0.0, 0.95, 0.05),
            (QUESTION, diagnosis): (0.0, 0.10, 0.90),
            ("Textbook passage.", fact): (0.0, 0.90, 0.10),
        }
    )
    base = assess_evidence([finding, diagnosis, fact], ["Textbook passage."], probs)

    grounded = ground_in_question(base, QUESTION, probs)

    by_text = {s.statement: s for s in grounded.statements}
    assert by_text[finding].from_question
    assert by_text[finding].supported
    assert not by_text[finding].literature_supported
    assert not by_text[diagnosis].supported  # an inference still needs the literature
    assert by_text[fact].literature_supported
    assert not is_grounded(grounded)  # the diagnosis is unsupported


def test_restating_the_question_alone_is_not_grounded() -> None:
    finding = "The patient is hypotensive with an irregular pulse."
    probs = table({(QUESTION, finding): (0.0, 0.95, 0.05)})
    only_question = ground_in_question(
        assess_evidence([finding], ["Unrelated passage."], probs), QUESTION, probs
    )

    assert only_question.statements[0].supported
    assert not is_grounded(only_question)
    assert present_answer("D", only_question, AnswerFormat.MULTIPLE_CHOICE) == INSUFFICIENT_EVIDENCE
