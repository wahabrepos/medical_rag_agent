"""Keyword tokenisation and rank fusion used by hybrid retrieval.

Both reproduce the research work: the BM25 tokenizer (lowercase, punctuation
stripped except hyphens inside words, English and medical-filler stopwords
removed) and Reciprocal Rank Fusion over the BM25 and dense result lists.
"""

import re
from collections.abc import Hashable, Sequence
from enum import StrEnum

RRF_K = 60

# English function words plus high-frequency medical filler words that carry no
# discriminative value for BM25. Medical entity terms are kept.
# fmt: off
STOPWORDS: frozenset[str] = frozenset(
    {
        # English function words
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "that", "this", "these", "those", "it", "its", "as", "not", "no",
        "nor", "so", "yet", "both", "either", "neither", "each", "than",
        "such", "up", "out", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "while", "where", "when", "who",
        "which", "their", "they", "them", "he", "she", "we", "our", "us",
        "also", "however", "therefore", "thus", "hence", "whereas",
        "moreover", "furthermore", "although", "because", "since", "whether",
        # High-frequency medical filler words
        "patient", "patients", "study", "studies", "result", "results",
        "method", "methods", "conclusion", "conclusions", "background",
        "objective", "purpose", "aim", "aims", "found", "showed", "shown",
        "using", "used", "based", "associated", "compared", "significantly",
        "among", "including", "included", "reported", "data", "analysis",
        "clinical", "case", "cases", "group", "groups", "effect", "effects",
    }
)
# fmt: on

_NON_WORD = re.compile(r"[^\w\s-]")
_LONE_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    BM25_ONLY = "bm25_only"
    DENSE_ONLY = "dense_only"


def tokenize(text: str) -> list[str]:
    """Tokens for BM25 indexing and querying.

    Hyphenated terms and codes such as "COVID-19", "TNF-alpha" or "HbA1c" stay intact.
    """
    text = _NON_WORD.sub(" ", text.lower())
    text = _LONE_HYPHEN.sub(" ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def reciprocal_rank_fusion[T: Hashable](
    bm25: Sequence[T],
    dense: Sequence[T],
    *,
    k: int = RRF_K,
    weight_bm25: float = 1.0,
    weight_dense: float = 1.0,
) -> list[T]:
    """Merge two ranked lists of document ids by weighted Reciprocal Rank Fusion.

    Each occurrence at rank r (1-based) adds weight / (k + r); an id that appears
    twice in one list is counted twice, as in the research work. Ties keep
    first-seen order (BM25 before dense). Equal weights give the research work's
    unweighted ordering.
    """
    scores: dict[T, float] = {}
    for weight, ranked in ((weight_bm25, bm25), (weight_dense, dense)):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def select_context[T: Hashable](
    bm25: Sequence[T],
    dense: Sequence[T],
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 5,
    k: int = RRF_K,
    weight_bm25: float = 1.0,
    weight_dense: float = 1.0,
) -> list[T]:
    """The passages handed to the generator for one iteration."""
    if mode is RetrievalMode.BM25_ONLY:
        return list(bm25[:top_k])
    if mode is RetrievalMode.DENSE_ONLY:
        return list(dense[:top_k])
    fused = reciprocal_rank_fusion(
        bm25, dense, k=k, weight_bm25=weight_bm25, weight_dense=weight_dense
    )
    return fused[:top_k]
