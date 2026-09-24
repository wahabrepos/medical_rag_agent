import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medrag_core.retrieval import tokenize
from medrag_search.bm25 import BM25Index

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("bm25_cases.json")


@pytest.fixture(scope="module")
def index() -> BM25Index:
    corpus = FIXTURE["corpus"]
    return BM25Index.from_texts(corpus, list(range(1000, 1000 + len(corpus))))


def test_average_idf_is_identical(index: BM25Index) -> None:
    assert index.average_idf == FIXTURE["average_idf"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["query"][:30] or "<empty>")
def test_scores_and_ranking_match_rank_bm25(index: BM25Index, case: dict[str, Any]) -> None:
    scores = index.scores(tokenize(case["query"]))
    top = index.top_positions(tokenize(case["query"]), 10)

    assert scores.tolist() == case["scores"]
    # Same ranking; documents with tied scores may come in a different order
    # (the research work's tie order depended on the CPU architecture).
    assert [scores[i] for i in top] == [scores[i] for i in case["top10"]]
    ranked = sorted(case["scores"], reverse=True)
    if len(set(ranked[:11])) == len(ranked[:11]):  # no ties in or at the edge of the top 10
        assert top == case["top10"]


def test_search_returns_external_ids(index: BM25Index) -> None:
    case = FIXTURE["cases"][0]
    assert index.search(case["query"], 3) == [1000 + p for p in case["top10"][:3]]


def test_save_and_load_round_trip(index: BM25Index, tmp_path: Path) -> None:
    path = tmp_path / "bm25.npz"
    index.save(path)
    loaded = BM25Index.load(path)

    for case in FIXTURE["cases"]:
        tokens = tokenize(case["query"])
        assert np.array_equal(loaded.scores(tokens), index.scores(tokens))
    assert loaded.vocab == index.vocab


def test_mismatched_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="one id per document"):
        BM25Index.build([["a"], ["b"]], [1])


def test_ties_are_broken_deterministically(index: BM25Index) -> None:
    top = index.top_positions([], 5)  # every document scores 0
    assert top == [index.size - 1 - i for i in range(5)]
