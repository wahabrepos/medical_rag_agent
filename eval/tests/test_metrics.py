import json
from pathlib import Path
from typing import Any

import pytest

from medrag_eval.datasets import load_manifest, load_reference_predictions
from medrag_eval.metrics import (
    Dataset,
    evaluate,
    extract_binary_answer,
    extract_option_letter,
    percentile,
)

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"
REFERENCE = Path(__file__).resolve().parents[1] / "reference"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


EXTRACT = load_fixture("extract_cases.json")
GOLDEN_METRICS = load_fixture("golden_metrics_cases.json")
STORED = json.loads((REFERENCE / "metrics.json").read_text("utf-8"))


def test_option_letter_extraction_matches_research_work() -> None:
    mismatches = [
        c["input"]
        for c in EXTRACT["option_letter"]
        if extract_option_letter(c["input"]) != c["expected"]
    ]
    assert len(EXTRACT["option_letter"]) > 100
    assert mismatches == []


def test_binary_extraction_matches_research_work() -> None:
    mismatches = [
        c["input"] for c in EXTRACT["binary"] if extract_binary_answer(c["input"]) != c["expected"]
    ]
    assert len(EXTRACT["binary"]) > 100
    assert mismatches == []


def _check(got: dict[str, float], expected: dict[str, float]) -> None:
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


@pytest.fixture(scope="module")
def manifest() -> dict[Dataset, list[dict[str, Any]]]:
    return load_manifest()


@pytest.mark.parametrize("case", GOLDEN_METRICS, ids=lambda c: f"{c['system']}-{c['dataset']}")
def test_golden_subset_metrics_match_research_work(
    case: dict[str, Any], manifest: dict[Dataset, list[dict[str, Any]]]
) -> None:
    dataset = Dataset(case["dataset"])
    golden_ids = {row["id"] for row in _golden_rows()}
    answers = {row["id"]: row["answer"] for row in manifest[dataset]}
    preds = [
        p
        for p in load_reference_predictions(case["system"])
        if p["id"] in golden_ids and p["id"] in answers
    ]

    got = evaluate(preds, [answers[p["id"]] for p in preds], dataset)

    assert len(preds) == case["n"]
    _check(got, case["expected"])


@pytest.mark.parametrize(
    ("system", "dataset"),
    [(s, d) for s, v in STORED["systems"].items() for d in v["datasets"]],
)
def test_full_set_metrics_match_research_work(
    system: str, dataset: str, manifest: dict[Dataset, list[dict[str, Any]]]
) -> None:
    ds = Dataset(dataset)
    answers = {row["id"]: row["answer"] for row in manifest[ds]}
    preds = [p for p in load_reference_predictions(system) if p["id"] in answers]

    got = evaluate(preds, [answers[p["id"]] for p in preds], ds)

    _check(got, STORED["systems"][system]["datasets"][dataset]["metrics"])


def test_percentile_interpolates_linearly() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0], 95) == 10.0


def _golden_rows() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "golden" / "golden_150.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
