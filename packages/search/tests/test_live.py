from typing import Any

import httpx
import pytest

from medrag_search.bm25 import BM25Index
from medrag_search.live import LiveAugmentedRetriever
from medrag_search.pubmed import (
    MAX_PASSAGE_CHARS,
    LiteratureSearchError,
    PubMedClient,
    parse_articles,
    truncate,
)
from medrag_search.retriever import Passage

EFETCH = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle><MedlineCitation><PMID>101</PMID><Article>
    <ArticleTitle>Metformin in <i>type 2</i> diabetes</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Metformin is widely used.</AbstractText>
      <AbstractText Label="RESULTS">It lowered HbA1c.</AbstractText>
    </Abstract></Article></MedlineCitation></PubmedArticle>
  <PubmedArticle><MedlineCitation><PMID>102</PMID><Article>
    <ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_articles_keeps_labels_and_markup_text() -> None:
    first, second = parse_articles(EFETCH)

    assert (first.pmid, first.chunk_id, first.document_id) == (101, -101, -1)
    assert first.title == "Metformin in type 2 diabetes"
    assert first.text == "BACKGROUND: Metformin is widely used. RESULTS: It lowered HbA1c."
    assert second.text == ""


def test_long_abstracts_are_cut_at_a_word_boundary() -> None:
    long = EFETCH.replace("It lowered HbA1c.", "word " * 5_000)
    first = parse_articles(long)[0]

    assert len(first.text) <= MAX_PASSAGE_CHARS
    assert first.text.endswith("word")
    assert truncate("short text", 100) == "short text"
    assert truncate("nospaces", 3) == "nos"


class FakeNcbi:
    def __init__(self, hits: dict[str, list[int]]) -> None:
        self.hits = hits
        self.terms: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params["tool"] == "medical_rag_agent"
        if request.url.path.endswith("esearch.fcgi"):
            self.terms.append(params["term"])
            ids = [str(i) for i in self.hits.get(params["term"], [])]
            return httpx.Response(200, json={"esearchresult": {"idlist": ids}})
        return httpx.Response(200, text=EFETCH)


def client(ncbi: Any) -> PubMedClient:
    transport = httpx.Client(transport=httpx.MockTransport(ncbi), base_url="https://eutils.test/")
    return PubMedClient(client=transport, sleep=lambda s: None)


def test_keywords_are_relaxed_until_pubmed_has_hits() -> None:
    ncbi = FakeNcbi({"metformin diabetes first": [101, 102]})
    found = client(ncbi).passages(["metformin", "diabetes", "first", "line"], 2)

    assert ncbi.terms == ["metformin diabetes first line", "metformin diabetes first"]
    assert [p.pmid for p in found] == [101]  # 102 has no abstract


def test_excluded_pmids_are_skipped() -> None:
    ncbi = FakeNcbi({"metformin diabetes first": [101]})
    found = client(ncbi).passages(["metformin", "diabetes", "first"], 2, exclude_pmids={101})
    assert found == []


def test_network_errors_raise() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(LiteratureSearchError):
        client(down).passages(["metformin", "diabetes", "first"], 2)


def test_bm25_keywords_prefer_rare_known_terms() -> None:
    index = BM25Index.from_texts(
        ["diabetes treatment common", "diabetes insulin", "diabetes metformin"], [1, 2, 3]
    )
    text = "A 63-year-old comes with diabetes treated by metformin or insulin?"

    # Known words only, rarest first; "diabetes" is in every document.
    assert index.keywords(text, 3) == ["metformin", "insulin", "diabetes"]
    assert index.keywords(text, 3, known_only=False)[0] == "comes"


def P(pmid: int) -> Passage:  # noqa: N802 - tiny test factory
    return Passage(chunk_id=pmid, document_id=pmid, pmid=pmid, title="", text=f"text {pmid}")


class FakeLocal:
    def __init__(self) -> None:
        self.bm25 = BM25Index.from_texts(["aspirin trial", "other study"], [1, 2])

    def passages(self, query: str, *, exclude_pmids: Any = ()) -> list[Passage]:
        return [P(i) for i in (1, 2, 3, 4, 5)]


class FakePubMed:
    def __init__(self, result: list[Passage] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[str], set[int]]] = []

    def passages(self, keywords: list[str], k: int, *, exclude_pmids: Any = ()) -> list[Passage]:
        self.calls.append((keywords, set(exclude_pmids)))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result[:k]


def test_live_results_replace_the_lowest_local_passages() -> None:
    pubmed = FakePubMed([P(-9), P(-8), P(-7)])
    retriever = LiveAugmentedRetriever(FakeLocal(), pubmed)  # type: ignore[arg-type]

    got = retriever.passages("Does aspirin help?\n\nAnswer choices:\nA. yes", exclude_pmids=[42])

    assert [p.pmid for p in got] == [1, 2, 3, -9, -8]
    keywords, excluded = pubmed.calls[0]
    assert "choices" not in keywords
    assert excluded == {42, 1, 2, 3, 4, 5}


def test_falls_back_to_local_when_pubmed_fails_or_finds_nothing() -> None:
    failing = LiveAugmentedRetriever(FakeLocal(), FakePubMed(LiteratureSearchError("down")))  # type: ignore[arg-type]
    empty = LiveAugmentedRetriever(FakeLocal(), FakePubMed([]))  # type: ignore[arg-type]

    assert [p.pmid for p in failing.passages("q")] == [1, 2, 3, 4, 5]
    assert empty("q") == [f"text {i}" for i in (1, 2, 3, 4, 5)]


def test_refinement_queries_search_for_the_claims() -> None:
    retriever = LiveAugmentedRetriever(FakeLocal(), FakePubMed([]))  # type: ignore[arg-type]
    query = "Other study words here? Find specific evidence for: aspirin trial outcome"

    terms = retriever.search_terms(query)

    assert "aspirin" in terms
    assert "other" not in terms
