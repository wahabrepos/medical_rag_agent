"""Background knowledge beyond PubMed abstracts, from files (research evaluation).

PubMed abstracts rarely state textbook facts, so clinical vignettes (MedQA) were
almost never grounded (Step 7b). A `KnowledgeIndex` is a file-based corpus with
the same retrieval method as the PubMed index (BM25 + BGE dense, rank fusion),
built by workers/ingest/gpu/export_textbooks.py and embed_chunks.py:

    <dir>/chunks.jsonl.gz     one snippet per line: id, corpus_order, sha256, book, title, text
    <dir>/vectors.npz         <profile>_vectors (float16, BGE [CLS], L2-normalised), ...
    <dir>/vectors.manifest.json
    <dir>/bm25.npz            built on first load

It is kept out of the database on purpose: the MedRAG textbooks have no licence
for products, so the corpus is only loaded when KNOWLEDGE_DIR is set.

`KnowledgeAugmentedRetriever` pools the PubMed passages with the best knowledge
snippets and keeps the ones closest to the query by BGE similarity (both corpora
use the same embedding model, so the scores are comparable).
"""

import gzip
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from medrag_core.retrieval import reciprocal_rank_fusion
from medrag_search.bm25 import BM25Index
from medrag_search.retriever import Passage

KNOWLEDGE_DOCUMENT_ID = -2
# Knowledge chunk ids are negative and far below any PubMed id (live passages use -pmid).
CHUNK_ID_OFFSET = 10**10

QueryEmbedder = Callable[[str], Sequence[float]]
TextEmbedder = Callable[[list[str]], npt.NDArray[np.float32]]


@dataclass(frozen=True)
class KnowledgePassage(Passage):
    """A snippet of a knowledge source; `pmid` is 0 (not a PubMed article)."""

    book: str = ""
    source: str = "textbook"


class KnowledgeIndex:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        vectors: npt.NDArray[np.float32],
        bm25: BM25Index,
        embed_query: QueryEmbedder,
    ) -> None:
        if len(rows) != len(vectors) or len(rows) != bm25.size:
            raise ValueError("chunks, vectors and BM25 index have different sizes")
        self.rows = rows
        self.vectors = vectors
        self.bm25 = bm25
        self.embed_query = embed_query

    @classmethod
    def load(cls, directory: Path, embed_query: QueryEmbedder) -> "KnowledgeIndex":
        with gzip.open(directory / "chunks.jsonl.gz", "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        with np.load(directory / "vectors.npz") as data:
            (profile,) = {k.rsplit("_", 1)[0] for k in data.files if k.endswith("_vectors")}
            order = data[f"{profile}_corpus_order"]
            vectors = np.empty((len(rows), data[f"{profile}_vectors"].shape[1]), np.float32)
            vectors[order] = data[f"{profile}_vectors"].astype(np.float32)
        bm25_path = directory / "bm25.npz"
        if bm25_path.exists():
            bm25 = BM25Index.load(bm25_path)
        else:
            bm25 = BM25Index.from_texts((r["text"] for r in rows), list(range(len(rows))))
            bm25.save(bm25_path)
        return cls(rows, vectors, bm25, embed_query)

    def passage(self, position: int) -> KnowledgePassage:
        row = self.rows[position]
        return KnowledgePassage(
            chunk_id=-(CHUNK_ID_OFFSET + position),
            document_id=KNOWLEDGE_DOCUMENT_ID,
            pmid=0,
            title=f"{row['book']}: {row['title']}",
            text=row["text"],
            book=row["book"],
        )

    def search(self, query: str, k: int = 5, *, candidates: int = 10) -> list[int]:
        """Positions of the k best snippets: BM25 and dense top-`candidates`, fused."""
        bm25 = self.bm25.search(query, candidates)
        q = np.asarray(self.embed_query(query), dtype=np.float32)
        scores = self.vectors @ q
        dense = [int(i) for i in np.argsort(-scores, kind="stable")[:candidates]]
        return reciprocal_rank_fusion(bm25, dense)[:k]


class PassageSource(Protocol):
    def passages(self, query: str, *, exclude_pmids: Sequence[int] = ...) -> list[Any]: ...


class KnowledgeAugmentedRetriever:
    """PubMed passages plus knowledge snippets, the `top_k` closest to the query."""

    def __init__(
        self,
        local: PassageSource,
        knowledge: KnowledgeIndex,
        embed_texts: TextEmbedder,
        *,
        top_k: int = 5,
        knowledge_k: int = 5,
    ) -> None:
        self.local = local
        self.knowledge = knowledge
        self.embed_texts = embed_texts
        self.top_k = top_k
        self.knowledge_k = knowledge_k

    def passages(self, query: str, *, exclude_pmids: Sequence[int] = ()) -> list[Passage]:
        local = list(self.local.passages(query, exclude_pmids=exclude_pmids))
        found = self.knowledge.search(query, self.knowledge_k)
        pool: list[Passage] = [*local, *(self.knowledge.passage(i) for i in found)]
        if not pool:
            return []
        q = np.asarray(self.knowledge.embed_query(query), dtype=np.float32)
        local_vectors = (
            self.embed_texts([p.text for p in local])
            if local
            else np.empty((0, len(q)), np.float32)
        )
        vectors = np.vstack([local_vectors, self.knowledge.vectors[found]])
        similarity = vectors @ q
        order = np.argsort(-similarity, kind="stable")[: self.top_k]
        return [pool[i] for i in order]

    def __call__(self, query: str, *, exclude_pmids: Sequence[int] = ()) -> list[str]:
        return [p.text for p in self.passages(query, exclude_pmids=exclude_pmids)]
