"""Hybrid retrieval: BM25 + dense search fused with Reciprocal Rank Fusion."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from medrag_core.retrieval import RRF_K, RetrievalMode, select_context
from medrag_db.models import Chunk, Document
from medrag_search.bm25 import BM25Index
from medrag_search.dense import dense_search

QueryEmbedder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    document_id: int
    pmid: int
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    bm25_ids: list[int]
    dense_ids: list[int]
    fused_ids: list[int]
    passages: list[Passage]

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.passages]


class HybridRetriever:
    """Research-work retrieval: top-10 from each retriever, RRF, top-5 passages."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        bm25: BM25Index,
        embed_query: QueryEmbedder,
        *,
        profile: str = "parity",
        mode: RetrievalMode = RetrievalMode.HYBRID,
        per_retriever_k: int = 10,
        top_k: int = 5,
        rrf_k: int = RRF_K,
        weight_bm25: float = 0.5,
        weight_dense: float = 0.5,
    ) -> None:
        self._sessions = session_factory
        self._bm25 = bm25
        self._embed = embed_query
        self.profile = profile
        self.mode = mode
        self.per_retriever_k = per_retriever_k
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense

    def search(self, query: str, *, exclude_pmids: Collection[int] = ()) -> RetrievalResult:
        """Hybrid search; chunks of the articles in `exclude_pmids` are never returned
        (used for leakage-free evaluation)."""
        with self._sessions() as session:
            excluded_docs: list[int] = []
            excluded_chunks: set[int] = set()
            if exclude_pmids:
                excluded_docs = list(
                    session.scalars(select(Document.id).where(Document.pmid.in_(exclude_pmids)))
                )
                excluded_chunks = set(
                    session.scalars(
                        select(Chunk.id).where(
                            Chunk.document_id.in_(excluded_docs), Chunk.profile == self.profile
                        )
                    )
                )
            bm25_ids = [
                i
                for i in self._bm25.search(query, self.per_retriever_k + len(excluded_chunks))
                if i not in excluded_chunks
            ][: self.per_retriever_k]
            dense_ids = dense_search(
                session,
                self._embed(query),
                profile=self.profile,
                k=self.per_retriever_k,
                exclude_document_ids=excluded_docs,
            )
            fused = select_context(
                bm25_ids,
                dense_ids,
                mode=self.mode,
                top_k=self.top_k,
                k=self.rrf_k,
                weight_bm25=self.weight_bm25,
                weight_dense=self.weight_dense,
            )
            passages = self._load(session, fused)
        return RetrievalResult(bm25_ids, dense_ids, fused, passages)

    def passages(self, query: str, *, exclude_pmids: Collection[int] = ()) -> list[Passage]:
        """Passages with their PubMed metadata (for citations)."""
        return self.search(query, exclude_pmids=exclude_pmids).passages

    def __call__(self, query: str, *, exclude_pmids: Collection[int] = ()) -> list[str]:
        """Passage texts, the shape the Self-MedRAG loop expects."""
        return self.search(query, exclude_pmids=exclude_pmids).texts

    @staticmethod
    def _load(session: Session, chunk_ids: Sequence[int]) -> list[Passage]:
        rows = session.execute(
            select(Chunk.id, Chunk.document_id, Document.pmid, Document.title, Chunk.text)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
        by_id = {r.id: Passage(r.id, r.document_id, r.pmid, r.title, r.text) for r in rows}
        return [by_id[i] for i in chunk_ids]
