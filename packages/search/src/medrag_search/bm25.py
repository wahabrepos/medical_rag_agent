"""BM25 (Okapi) keyword index, numerically identical to the research work's `rank_bm25`.

Scores are computed with the same operations in the same order as
`rank_bm25.BM25Okapi` 0.2.2 (including the epsilon floor for negative IDF), so
scores match bit for bit and rankings match wherever scores are not tied.
Postings are stored per term, so a query only touches the documents that contain
its terms when building the frequency vectors.
"""

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from medrag_core.retrieval import tokenize

FORMAT_VERSION = 1


@dataclass
class BM25Index:
    doc_ids: npt.NDArray[np.int64]
    """External id (chunk id) of each indexed document, in index order."""
    vocab: dict[str, int]
    idf: npt.NDArray[np.float64]
    term_ptr: npt.NDArray[np.int64]
    post_docs: npt.NDArray[np.int64]
    post_tfs: npt.NDArray[np.int64]
    doc_len: npt.NDArray[np.int64]
    avgdl: float
    average_idf: float
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        # Same expression rank_bm25 evaluates for every query term.
        self._norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @classmethod
    def build(
        cls,
        tokenized: Iterable[Sequence[str]],
        doc_ids: Sequence[int],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> "BM25Index":
        nd: dict[str, int] = {}
        postings: dict[str, tuple[list[int], list[int]]] = {}
        doc_len: list[int] = []
        num_doc = 0
        corpus_size = 0
        for pos, document in enumerate(tokenized):
            doc_len.append(len(document))
            num_doc += len(document)
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
                docs, tfs = postings.setdefault(word, ([], []))
                docs.append(pos)
                tfs.append(freq)
            corpus_size += 1
        if corpus_size != len(doc_ids):
            raise ValueError("doc_ids must have one id per document")

        vocab = {word: i for i, word in enumerate(nd)}
        idf = np.empty(len(vocab), dtype=np.float64)
        idf_sum = 0.0
        negative: list[int] = []
        for word, freq in nd.items():
            value = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[vocab[word]] = value
            idf_sum += value
            if value < 0:
                negative.append(vocab[word])
        average_idf = idf_sum / len(vocab) if vocab else 0.0
        idf[negative] = epsilon * average_idf

        term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        docs_flat: list[int] = []
        tfs_flat: list[int] = []
        for word, i in vocab.items():
            d, t = postings[word]
            docs_flat.extend(d)
            tfs_flat.extend(t)
            term_ptr[i + 1] = len(docs_flat)

        return cls(
            doc_ids=np.asarray(doc_ids, dtype=np.int64),
            vocab=vocab,
            idf=idf,
            term_ptr=term_ptr,
            post_docs=np.asarray(docs_flat, dtype=np.int64),
            post_tfs=np.asarray(tfs_flat, dtype=np.int64),
            doc_len=np.asarray(doc_len, dtype=np.int64),
            avgdl=num_doc / corpus_size,
            average_idf=average_idf,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], doc_ids: Sequence[int], **kwargs: float
    ) -> "BM25Index":
        return cls.build((tokenize(t) for t in texts), doc_ids, **kwargs)

    def scores(self, query_tokens: Sequence[str]) -> npt.NDArray[np.float64]:
        score = np.zeros(self.size)
        for q in query_tokens:
            q_freq = np.zeros(self.size, dtype=np.int64)
            term = self.vocab.get(q)
            idf: float = 0.0
            if term is not None:
                start, end = self.term_ptr[term], self.term_ptr[term + 1]
                q_freq[self.post_docs[start:end]] = self.post_tfs[start:end]
                idf = float(self.idf[term])
            score += idf * (q_freq * (self.k1 + 1) / (q_freq + self._norm))
        return score

    def top_positions(self, query_tokens: Sequence[str], n: int) -> list[int]:
        """Index positions of the n best documents, best first.

        rank_bm25 uses numpy's default sort, whose order for tied scores differs
        between CPU architectures. A stable sort makes ties deterministic on every
        platform (the later document first); non-tied rankings are unchanged.
        """
        order = np.argsort(self.scores(query_tokens), kind="stable")[::-1][:n]
        return [int(i) for i in order]

    def search(self, query: str, n: int = 10) -> list[int]:
        """External ids of the n best documents for a raw query string."""
        return [int(self.doc_ids[i]) for i in self.top_positions(tokenize(query), n)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "avgdl": self.avgdl,
            "average_idf": self.average_idf,
            "vocab": list(self.vocab),
        }
        with path.open("wb") as fh:
            np.savez_compressed(
                fh,
                meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
                doc_ids=self.doc_ids,
                idf=self.idf,
                term_ptr=self.term_ptr,
                post_docs=self.post_docs,
                post_tfs=self.post_tfs,
                doc_len=self.doc_len,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with np.load(path) as data:
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
            if meta["format"] != FORMAT_VERSION:
                raise ValueError(f"unsupported BM25 index format {meta['format']}")
            return cls(
                doc_ids=data["doc_ids"],
                vocab={word: i for i, word in enumerate(meta["vocab"])},
                idf=data["idf"],
                term_ptr=data["term_ptr"],
                post_docs=data["post_docs"],
                post_tfs=data["post_tfs"],
                doc_len=data["doc_len"],
                avgdl=meta["avgdl"],
                average_idf=meta["average_idf"],
                k1=meta["k1"],
                b=meta["b"],
                epsilon=meta["epsilon"],
            )
