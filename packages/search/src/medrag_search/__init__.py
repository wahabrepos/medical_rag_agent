"""Hybrid retrieval: BM25 index, pgvector search and rank fusion."""

from medrag_search.bm25 import BM25Index
from medrag_search.dense import dense_search
from medrag_search.retriever import HybridRetriever, Passage, RetrievalResult

__all__ = ["BM25Index", "HybridRetriever", "Passage", "RetrievalResult", "dense_search"]
__version__ = "0.1.0"
