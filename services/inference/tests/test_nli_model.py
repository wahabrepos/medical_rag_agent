"""NLI against the research work's PyTorch outputs (downloads the model; run with -m model)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medrag_core.verification import verify_rationale
from medrag_inference.nli import LABELS, RESEARCH_WORK_SUPPORT_LABEL, DebertaNli

pytestmark = pytest.mark.model

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("nli_cases.json")
PAIRS = [(p, s) for c in FIXTURE["cases"] for s in c["rationale"] for p in c["passages"]]
REFERENCE = np.array([row for c in FIXTURE["cases"] for row in c["probabilities"]])
THETA = FIXTURE["verification_threshold"]


@pytest.fixture(scope="module")
def nli() -> DebertaNli:
    return DebertaNli()


def test_label_order_matches_model_config() -> None:
    assert [FIXTURE["id2label"][str(i)] for i in range(3)] == list(LABELS)
    assert LABELS[FIXTURE["research_work_column"]] == RESEARCH_WORK_SUPPORT_LABEL


def test_matches_research_work_pytorch(nli: DebertaNli) -> None:
    assert np.abs(nli.probabilities(PAIRS) - REFERENCE).max() < 1e-3


def test_research_work_support_scores_are_reproduced(nli: DebertaNli) -> None:
    scorer = nli.scorer(RESEARCH_WORK_SUPPORT_LABEL)
    for case in FIXTURE["cases"]:
        got = verify_rationale(case["rationale"], case["passages"], scorer, threshold=THETA)
        assert got.support_score == case["research_work_support_score"], case["id"]
        assert got.unsupported == case["research_work_unsupported"], case["id"]


def test_cache_returns_identical_scores_without_recomputing(nli: DebertaNli) -> None:
    first = nli.probabilities(PAIRS[:10])
    runs = 0
    original = nli._compute

    def counting(pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
        nonlocal runs
        runs += 1
        return original(pairs, batch_size)

    nli._compute = counting  # type: ignore[method-assign]
    again = nli.probabilities(PAIRS[:10])

    assert runs == 0
    assert np.array_equal(first, again)
