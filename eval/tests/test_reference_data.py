"""Consistency checks for the research-work reference data captured in Step 2."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

EVAL = Path(__file__).resolve().parents[1]
REFERENCE = EVAL / "reference"
SYSTEMS = (
    "base_bm25",
    "base_dense",
    "base_hybrid",
    "selfmedrag_qwen_full",
    "selfmedrag_qwen7b",
    "selfmedrag_qwen14b",
    "selfmedrag_mistral",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def manifest() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, Any] = json.loads((REFERENCE / "eval_sets_manifest.json").read_text("utf-8"))
    return {name: data[name] for name in ("medqa", "pubmedqa")}


def test_manifest_sizes_and_unique_ids(manifest: dict[str, list[dict[str, Any]]]) -> None:
    assert len(manifest["medqa"]) == 1000
    assert len(manifest["pubmedqa"]) == 890
    ids = [row["id"] for rows in manifest.values() for row in rows]
    assert len(ids) == len(set(ids))


def test_manifest_answers(manifest: dict[str, list[dict[str, Any]]]) -> None:
    assert {row["answer"] for row in manifest["medqa"]} <= set("ABCDE")
    assert {row["answer"] for row in manifest["pubmedqa"]} == {"yes", "no"}


def test_golden_subset_is_drawn_from_manifest(manifest: dict[str, list[dict[str, Any]]]) -> None:
    golden = _jsonl(EVAL / "golden" / "golden_150.jsonl")
    by_id = {row["id"]: row for rows in manifest.values() for row in rows}

    assert Counter(row["dataset"] for row in golden) == {"medqa": 75, "pubmedqa": 75}
    for row in golden:
        ref = by_id[row["id"]]
        assert row["answer"] == ref["answer"]
        assert "selfmedrag_mistral" in row["reference"]


@pytest.mark.parametrize("system", SYSTEMS)
def test_predictions_follow_manifest_order(
    system: str, manifest: dict[str, list[dict[str, Any]]]
) -> None:
    rows = _jsonl(REFERENCE / "predictions" / f"{system}.jsonl")
    expected = [r["id"] for name in ("medqa", "pubmedqa") for r in manifest[name]]
    got = [r["id"] for r in rows]
    if system == "base_hybrid":  # only PubMedQA was stored for this baseline
        expected = [r["id"] for r in manifest["pubmedqa"]]
    assert got == expected


def test_parity_targets_match_stored_metrics() -> None:
    metrics = json.loads((REFERENCE / "metrics.json").read_text("utf-8"))
    targets = json.loads((REFERENCE / "parity_settings.json").read_text("utf-8"))["targets"]
    parity = metrics["systems"]["selfmedrag_mistral"]["datasets"]

    assert parity["medqa"]["metrics"]["accuracy"] == pytest.approx(targets["medqa_accuracy"])
    assert parity["pubmedqa"]["metrics"]["accuracy"] == pytest.approx(
        targets["pubmedqa_accuracy"], abs=1e-4
    )


def test_retrieval_reference_covers_200_questions(
    manifest: dict[str, list[dict[str, Any]]],
) -> None:
    rows = _jsonl(REFERENCE / "retrieval_200.jsonl")
    known = {row["id"] for rows_ in manifest.values() for row in rows_}

    assert len(rows) == 200
    assert {row["id"] for row in rows} <= known
    assert all(len(row["fused_top5"]) == 5 for row in rows)
    assert all(len(row["bm25_top10"]) == 10 for row in rows)
