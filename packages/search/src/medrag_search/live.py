"""Hybrid retrieval topped up with live PubMed results.

The local corpus (PubMedQA abstracts) lacks much clinical knowledge; PubMed has
far more. For each query the best local passages are kept and the lowest-ranked
ones are replaced by the best live abstracts not already present, so prompt size
and NLI cost stay the same. If PubMed cannot be reached, local results are used.
"""

import logging
from collections.abc import Collection

from medrag_search.pubmed import LiteratureSearchError, PubMedClient
from medrag_search.retriever import HybridRetriever, Passage

logger = logging.getLogger(__name__)

MCQ_OPTIONS_MARKER = "\n\nAnswer choices:"
# medrag_core.policy's structured refinement: "<question> Find specific evidence for: <claims>"
REFINEMENT_MARKER = "Find specific evidence for:"


class LiveAugmentedRetriever:
    def __init__(
        self,
        local: HybridRetriever,
        pubmed: PubMedClient,
        *,
        live_k: int = 2,
        keyword_count: int = 5,
    ) -> None:
        self.local = local
        self.pubmed = pubmed
        self.live_k = live_k
        self.keyword_count = keyword_count

    def search_terms(self, query: str) -> list[str]:
        """Keywords for PubMed, rarest known words first.

        In a refinement query the unsupported claims name the medical concepts to look
        up, so only they are used; clinical vignettes themselves are mostly everyday
        words that make a literature search fail. Otherwise the question (without
        its answer options) is used.
        """
        if REFINEMENT_MARKER in query:
            text = query.rsplit(REFINEMENT_MARKER, 1)[1]
        else:
            text = query.split(MCQ_OPTIONS_MARKER, 1)[0]
        return self.local.bm25.keywords(text, self.keyword_count)

    def passages(self, query: str, *, exclude_pmids: Collection[int] = ()) -> list[Passage]:
        local = self.local.passages(query, exclude_pmids=exclude_pmids)
        skip = set(exclude_pmids) | {p.pmid for p in local}
        try:
            live = self.pubmed.passages(self.search_terms(query), self.live_k, exclude_pmids=skip)
        except LiteratureSearchError as exc:
            logger.warning("live PubMed search failed, using local results: %s", exc)
            return local
        if not live:
            return local
        return [*local[: max(len(local) - len(live), 0)], *live]

    def __call__(self, query: str, *, exclude_pmids: Collection[int] = ()) -> list[str]:
        return [p.text for p in self.passages(query, exclude_pmids=exclude_pmids)]
