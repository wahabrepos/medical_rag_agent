import gzip
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from medrag_search.knowledge import (
    KnowledgeAugmentedRetriever,
    KnowledgeIndex,
    KnowledgePassage,
)
from medrag_search.retriever import Passage

TEXTS = [
    "Thiamine deficiency causes Wernicke encephalopathy with confusion and ataxia.",
    "Aspirin irreversibly inhibits cyclooxygenase in platelets.",
    "Metformin decreases hepatic gluconeogenesis.",
]
# Unit vectors; the fake embedder maps text to the axis of its first matching keyword.
AXES = {"thiamine": 0, "aspirin": 1, "metformin": 2, "asthma": 3}


def embed_one(text: str) -> npt.NDArray[np.float32]:
    v = np.zeros(4, np.float32)
    for word, axis in AXES.items():
        if word in text.lower():
            v[axis] = 1.0
            break
    return v


def embed_query(text: str) -> list[float]:
    return [float(x) for x in embed_one(text)]


def embed_texts(texts: list[str]) -> npt.NDArray[np.float32]:
    return np.vstack([embed_one(t) for t in texts])


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    with gzip.open(tmp_path / "chunks.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i, text in enumerate(TEXTS):
            row = {
                "id": f"b_{i}",
                "corpus_order": i,
                "book": "Book",
                "title": f"T{i}",
                "text": text,
            }
            fh.write(json.dumps(row) + "\n")
    order = np.array([2, 0, 1])  # stored out of order on purpose
    np.savez(
        tmp_path / "vectors.npz",
        textbooks_corpus_order=order,
        textbooks_vectors=embed_texts([TEXTS[i] for i in order]).astype(np.float16),
    )
    return tmp_path


def test_load_search_and_cached_bm25(index_dir: Path) -> None:
    index = KnowledgeIndex.load(index_dir, embed_query)

    assert index.search("thiamine deficiency", k=1) == [0]
    assert np.array_equal(index.vectors[1], embed_one(TEXTS[1]))  # reordered by corpus_order
    assert (index_dir / "bm25.npz").exists()
    assert KnowledgeIndex.load(index_dir, embed_query).search("aspirin", k=1) == [1]

    passage = index.passage(0)
    assert isinstance(passage, KnowledgePassage)
    assert (passage.pmid, passage.source, passage.title) == (0, "textbook", "Book: T0")
    assert passage.chunk_id < -(10**9)  # never collides with live PubMed ids (-pmid)


class FakeLocal:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.excluded: object = None

    def passages(self, query: str, *, exclude_pmids: object = ()) -> list[Passage]:
        self.excluded = exclude_pmids
        return [Passage(i, i, 1000 + i, "PubMed", t) for i, t in enumerate(self.texts)]


def test_pool_keeps_the_closest_passages(index_dir: Path) -> None:
    local = FakeLocal(["Asthma trial of inhaled steroids.", "Thiamine levels in alcoholics."])
    retriever = KnowledgeAugmentedRetriever(
        local, KnowledgeIndex.load(index_dir, embed_query), embed_texts, top_k=2
    )

    found = retriever.passages("thiamine and confusion", exclude_pmids=[7])

    assert local.excluded == [7]
    assert {p.text for p in found} == {TEXTS[0], "Thiamine levels in alcoholics."}
    assert retriever("thiamine") == [p.text for p in retriever.passages("thiamine")]


def test_mismatched_files_are_rejected(index_dir: Path) -> None:
    index = KnowledgeIndex.load(index_dir, embed_query)
    with pytest.raises(ValueError, match="different sizes"):
        KnowledgeIndex(index.rows[:2], index.vectors, index.bm25, embed_query)
