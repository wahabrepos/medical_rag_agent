import numpy as np
import pytest
from sqlalchemy import text

from medrag_db import Chunk, Document, make_engine, make_session_factory
from medrag_db.indexes import create_hnsw_index_sql, drop_hnsw_index_sql
from medrag_search import BM25Index, HybridRetriever, dense_search

pytestmark = pytest.mark.integration

TEXTS = [
    "Aspirin lowers the risk of myocardial infarction.",
    "Metformin is first-line therapy for type 2 diabetes.",
    "Statins reduce LDL cholesterol.",
    "Insulin is needed in type 1 diabetes.",
]


def _unit(i: int) -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture(scope="module")
def seeded(database_url: str) -> list[int]:
    engine = make_engine(database_url)
    sessions = make_session_factory(engine)
    ids = []
    with sessions() as session, session.begin():
        for i, t in enumerate(TEXTS):
            doc = Document(pmid=1000 + i, source="test", title=f"title {i}", meta={})
            session.add(doc)
            session.flush()
            chunk = Chunk(
                document_id=doc.id,
                profile="parity",
                corpus_order=i,
                text=t,
                sha256=str(i),
                embedding=_unit(i),
            )
            session.add(chunk)
            session.flush()
            ids.append(chunk.id)
    with engine.begin() as conn:
        conn.execute(text(drop_hnsw_index_sql("parity")))
        conn.execute(text(create_hnsw_index_sql("parity")))
    return ids


def test_dense_search_orders_by_cosine(database_url: str, seeded: list[int]) -> None:
    query = _unit(2) * 0.9 + _unit(0) * 0.1
    with make_session_factory(make_engine(database_url))() as session:
        got = dense_search(session, query.tolist(), profile="parity", k=2)
        other = dense_search(session, query.tolist(), profile="standard", k=2)

    assert got == [seeded[2], seeded[0]]
    assert other == []


def test_hybrid_retriever_returns_passages(database_url: str, seeded: list[int]) -> None:
    bm25 = BM25Index.from_texts(TEXTS, seeded)
    retriever = HybridRetriever(
        make_session_factory(make_engine(database_url)),
        bm25,
        lambda _q: _unit(1).tolist(),
        top_k=2,
    )

    result = retriever.search("metformin diabetes")

    assert result.bm25_ids[0] == seeded[1]
    assert result.dense_ids[0] == seeded[1]
    assert result.fused_ids[0] == seeded[1]
    assert result.passages[0].pmid == 1001
    assert result.passages[0].text == TEXTS[1]
    assert retriever("metformin diabetes") == result.texts


def test_excluded_articles_are_never_returned(database_url: str, seeded: list[int]) -> None:
    bm25 = BM25Index.from_texts(TEXTS, seeded)
    retriever = HybridRetriever(
        make_session_factory(make_engine(database_url)),
        bm25,
        lambda _q: _unit(1).tolist(),
        top_k=3,
    )

    result = retriever.search("metformin diabetes", exclude_pmids=[1001])

    assert seeded[1] not in result.bm25_ids + result.dense_ids + result.fused_ids
    assert all(p.pmid != 1001 for p in result.passages)
    assert len(result.fused_ids) == 3  # still a full top-k from the other articles
