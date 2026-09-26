"""Live PubMed search through NCBI E-utilities, as a fallback evidence source.

    esearch (relevance-sorted PubMed IDs) -> efetch (titles and abstracts)

Requests are throttled to NCBI's limit (3/s, or 10/s with an API key) and cached.
Failures raise LiteratureSearchError; callers treat live search as optional.
Live results are Passage objects with chunk_id = -pmid and document_id = -1.
"""

import threading
import time
from collections.abc import Callable, Collection
from functools import lru_cache
from typing import Any

import httpx
from defusedxml import ElementTree as SafeET

from medrag_search.retriever import Passage

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
LIVE_DOCUMENT_ID = -1
# Longer than the NLI model reads (512 tokens) and than any research-work chunk;
# some records (meeting-abstract collections) run to tens of thousands of characters.
MAX_PASSAGE_CHARS = 4_000


class LiteratureSearchError(RuntimeError):
    """PubMed could not be searched (network, NCBI error)."""


def live_chunk_id(pmid: int) -> int:
    return -pmid


class PubMedClient:
    def __init__(
        self,
        *,
        tool: str = "medical_rag_agent",
        email: str | None = None,
        api_key: str | None = None,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._params: dict[str, str] = {"tool": tool}
        if email:
            self._params["email"] = email
        if api_key:
            self._params["api_key"] = api_key
        self._interval = 0.11 if api_key else 0.34  # 10 or 3 requests per second
        self._client = client or httpx.Client(base_url=EUTILS, timeout=timeout)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last = 0.0
        self.search = lru_cache(maxsize=2048)(self._search)
        self._fetch_one = lru_cache(maxsize=8192)(self._fetch_uncached)

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        with self._lock:
            wait = self._last + self._interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
        try:
            response = self._client.get(path, params={**self._params, **params})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LiteratureSearchError(f"PubMed {path} failed: {exc}") from exc
        return response

    def _search(self, term: str, max_results: int = 10) -> tuple[int, ...]:
        body: dict[str, Any] = self._get(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": term,
                "retmax": str(max_results),
                "sort": "relevance",
                "retmode": "json",
            },
        ).json()
        return tuple(int(i) for i in body.get("esearchresult", {}).get("idlist", []))

    def _fetch_uncached(self, pmids: tuple[int, ...]) -> tuple[Passage, ...]:
        xml = self._get(
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(map(str, pmids)), "retmode": "xml"},
        ).text
        return tuple(parse_articles(xml))

    def fetch(self, pmids: Collection[int]) -> list[Passage]:
        return list(self._fetch_one(tuple(pmids))) if pmids else []

    def passages(
        self,
        keywords: list[str],
        k: int,
        *,
        exclude_pmids: Collection[int] = (),
        min_terms: int = 3,
    ) -> list[Passage]:
        """Up to k abstracts for the keywords, relaxing from all of them (AND) down to
        `min_terms`; excluded PubMed IDs are skipped."""
        excluded = set(exclude_pmids)
        for size in range(len(keywords), min_terms - 1, -1):
            ids = [
                i
                for i in self.search(" ".join(keywords[:size]), k + len(excluded))
                if i not in excluded
            ]
            if ids:
                found = [p for p in self.fetch(ids[:k]) if p.text]
                if found:
                    return found
        return []


def truncate(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters, at a word boundary when there is one."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def parse_articles(xml: str) -> list[Passage]:
    """Title and abstract of every PubmedArticle; labelled sections keep their label."""
    passages = []
    # PubMed XML comes from the network: parse it without entity expansion.
    for article in SafeET.fromstring(xml).iter("PubmedArticle"):
        pmid_text = article.findtext(".//MedlineCitation/PMID")
        if not pmid_text:
            continue
        title = (
            "".join(article.find(".//ArticleTitle").itertext())
            if article.find(".//ArticleTitle") is not None
            else ""
        )
        parts = []
        for node in article.iterfind(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")
            if text:
                parts.append(f"{label}: {text}" if label else text)
        pmid = int(pmid_text)
        passages.append(
            Passage(
                chunk_id=live_chunk_id(pmid),
                document_id=LIVE_DOCUMENT_ID,
                pmid=pmid,
                title=title.strip(),
                text=truncate(" ".join(parts), MAX_PASSAGE_CHARS),
            )
        )
    return passages
