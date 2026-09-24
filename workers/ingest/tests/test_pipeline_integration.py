import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from sqlalchemy import func, select

from medrag_db import Chunk, Document, Section, make_engine
from medrag_db.session import make_session_factory
from medrag_ingest import pipeline
from medrag_ingest.corpus import SectionRecord
from medrag_search import BM25Index

pytestmark = pytest.mark.integration

WORDS = " ".join(f"word{i}" for i in range(30))


class FakeEmbedder:
    model_id = "fake/model"
    version = "fake@1"

    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Sequence[str], *, batch_size: int = 8) -> npt.NDArray[np.float32]:
        self.calls += 1
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for row, t in enumerate(texts):
            out[row, int(hashlib.sha256(t.encode()).hexdigest(), 16) % 384] = 1.0
        return out


RECORDS = [
    SectionRecord(0, 111, "pqa_labeled", "Q1?", 0, "BACKGROUND", f"Alpha. {WORDS}", {}),
    SectionRecord(1, 111, "pqa_labeled", "Q1?", 1, "RESULTS", f"Beta. {WORDS}", {}),
    SectionRecord(2, 222, "pqa_unlabeled", "Q2?", 0, None, "Too short to keep.", {}),
    SectionRecord(3, 222, "pqa_unlabeled", "Q2?", 2, "METHODS", f"Gamma. {WORDS}", {}),
]


def test_full_pipeline(database_url: str, tmp_path: Path) -> None:
    engine = make_engine(database_url)
    embedder = FakeEmbedder()

    report = pipeline.load_corpus(engine, RECORDS, count_tokens=embedder.count_tokens)
    embedded = {p: pipeline.embed_missing(engine, embedder, profile=p) for p in report.chunks}
    resumed = pipeline.embed_missing(engine, embedder, profile="parity")
    for profile in report.chunks:
        pipeline.build_vector_index(engine, profile=profile)
    bm25 = pipeline.build_bm25(engine, profile="parity", path=tmp_path / "bm25.npz")

    assert (report.documents, report.sections) == (2, 4)
    # Research-work chunking drops the under-20-word section; standard keeps whole abstracts.
    assert report.chunks == {"parity": 3, "standard": 2}
    assert embedded == {"parity": 3, "standard": 2}
    assert resumed == 0
    assert BM25Index.load(tmp_path / "bm25.npz").size == bm25.size == 3

    with make_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).where(Chunk.embedding.is_(None))) == 0
        assert (
            session.scalar(select(Section.label).where(Section.corpus_position == 1)) == "RESULTS"
        )
        assert session.scalars(select(Document.pmid).order_by(Document.pmid)).all() == [111, 222]
        standard = session.scalars(
            select(Chunk.text).where(Chunk.profile == "standard").order_by(Chunk.corpus_order)
        ).all()
    assert standard[0].startswith("BACKGROUND: Alpha.")


def test_reset_empties_corpus(database_url: str) -> None:
    engine = make_engine(database_url)
    pipeline.reset_corpus(engine)
    with make_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Chunk)) == 0
