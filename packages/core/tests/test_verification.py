import json
from pathlib import Path
from typing import Any

import pytest

from medrag_core.verification import verify_rationale

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("verification_cases.json")


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_verification_matches_research_work(case: dict[str, Any]) -> None:
    seen: list[list[str]] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        seen.extend([list(p) for p in pairs])
        return list(case["entailment_probs"])

    result = verify_rationale(
        case["rationale"],
        case["context"],
        scorer,
        threshold=FIXTURE["verification_threshold"],
    )

    assert seen == case["expected_pairs"]
    assert result.support_score == case["expected_support_score"]
    assert result.unsupported == case["expected_unsupported"]


def test_normalized_verification_checks_claims() -> None:
    seen: list[tuple[str, str]] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        seen.extend(pairs)
        return [0.9 if h == "Aspirin helps." else 0.1 for _, h in pairs]

    result = verify_rationale(
        ["Passage 1 shows that aspirin helps.", "Passage 2 states that X is rare."],
        ["p1"],
        scorer,
        normalize=True,
    )

    assert [h for _, h in seen] == ["Aspirin helps.", "X is rare."]
    assert result.support_score == 0.5
    assert result.unsupported == ["X is rare."]  # normalised, so refinement queries are clean


def _probs(rows: dict[tuple[str, str], tuple[float, float, float]]):  # type: ignore[no-untyped-def]
    return lambda pairs: [rows.get(p, (0.0, 0.1, 0.9)) for p in pairs]


def test_contradiction_vetoes_support() -> None:
    from medrag_core.verification import verify_with_contradictions

    probs = _probs({("p1", "A helps."): (0.0, 0.9, 0.1), ("p2", "B is safe."): (0.8, 0.1, 0.1)})
    result = verify_with_contradictions(["A helps.", "B is safe."], ["p1", "p2"], probs)

    assert result.support_score == 0.0
    assert result.unsupported == ["B is safe."]
    assert result.best_scores == [0.9, 0.1]


def test_without_contradiction_support_is_the_entailed_share() -> None:
    from medrag_core.verification import verify_with_contradictions

    probs = _probs({("p1", "A helps."): (0.0, 0.9, 0.1)})
    result = verify_with_contradictions(["Passage 1 states that A helps.", "Other."], ["p1"], probs)
    assert result.support_score == 0.5
    assert result.unsupported == ["Other."]


def test_weak_contradiction_does_not_veto() -> None:
    from medrag_core.verification import verify_with_contradictions

    probs = _probs({("p1", "A helps."): (0.75, 0.2, 0.05), ("p2", "A helps."): (0.0, 0.8, 0.2)})
    assert verify_with_contradictions(["A helps."], ["p1", "p2"], probs).support_score == 1.0
