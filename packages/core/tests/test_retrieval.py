import json
from pathlib import Path
from typing import Any

import pytest

from medrag_core.retrieval import (
    RetrievalMode,
    reciprocal_rank_fusion,
    select_context,
    tokenize,
)

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


RRF = load_fixture("rrf_cases.json")
TOKENS = load_fixture("bm25_tokenizer_cases.json")


def test_rrf_constant() -> None:
    assert RRF["rrf_k"] == 60


@pytest.mark.parametrize("case", RRF["cases"], ids=lambda c: c["id"])
def test_rrf_matches_research_work(case: dict[str, Any]) -> None:
    assert reciprocal_rank_fusion(case["bm25"], case["dense"], k=60) == case["expected"]


@pytest.mark.parametrize("case", RRF["cases"], ids=lambda c: c["id"])
def test_equal_weights_keep_research_work_order(case: dict[str, Any]) -> None:
    fused = reciprocal_rank_fusion(
        case["bm25"], case["dense"], k=60, weight_bm25=0.5, weight_dense=0.5
    )
    assert fused == case["expected"]


def test_weights_change_the_order() -> None:
    bm25, dense = ["a", "b"], ["b", "a"]

    assert reciprocal_rank_fusion(bm25, dense, weight_bm25=2.0)[0] == "a"
    assert reciprocal_rank_fusion(bm25, dense, weight_dense=2.0)[0] == "b"


def test_select_context_modes() -> None:
    bm25, dense = ["a", "b", "c"], ["c", "d"]

    assert select_context(bm25, dense, mode=RetrievalMode.BM25_ONLY, top_k=2) == ["a", "b"]
    assert select_context(bm25, dense, mode=RetrievalMode.DENSE_ONLY, top_k=5) == ["c", "d"]
    assert select_context(bm25, dense, top_k=2) == reciprocal_rank_fusion(bm25, dense)[:2]


@pytest.mark.parametrize("case", TOKENS, ids=lambda c: c["text"][:40])
def test_tokenizer_matches_research_work(case: dict[str, Any]) -> None:
    assert tokenize(case["text"]) == case["tokens"]
