import numpy as np
import pytest
from sqlalchemy import inspect, select, text

from medrag_db import Chunk, Document, Section, make_engine, make_session_factory
from medrag_db.testing import downgrade, migrate

pytestmark = pytest.mark.integration


def test_migration_creates_schema(database_url: str) -> None:
    engine = make_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    indexes = {i["name"] for i in inspect(engine).get_indexes("chunks")}

    assert {"documents", "sections", "chunks", "runs", "feedback"} <= tables
    assert {"ix_chunks_embedding_hnsw_parity", "ix_chunks_embedding_hnsw_standard"} <= indexes
    assert "ix_chunks_tsv" in indexes


def test_round_trip_with_halfvec_and_tsvector(database_url: str) -> None:
    sessions = make_session_factory(make_engine(database_url))
    vector = np.linspace(-1, 1, 384, dtype=np.float32)
    with sessions() as session, session.begin():
        doc = Document(pmid=12345, source="pqa_labeled", title="Q?", meta={"meshes": ["X"]})
        section = Section(document=doc, position=0, label="RESULTS", text="t", corpus_position=0)
        session.add(section)
        session.flush()
        session.add(
            Chunk(
                document_id=doc.id,
                section_id=section.id,
                profile="parity",
                corpus_order=0,
                text="Aspirin reduces cardiovascular events.",
                sha256="0" * 64,
                embedding=vector,
            )
        )

    with sessions() as session:
        chunk = session.scalars(select(Chunk)).one()
        tsv = session.scalar(text("select tsv::text from chunks"))

    assert np.allclose(np.asarray(chunk.embedding), vector, atol=1e-3)  # half precision
    assert "aspirin" in tsv
    assert "cardiovascular" in tsv


def test_profile_is_constrained(database_url: str) -> None:
    sessions = make_session_factory(make_engine(database_url))
    with pytest.raises(Exception, match="chunk_profile"), sessions() as session, session.begin():
        session.execute(
            text(
                "insert into chunks (document_id, profile, corpus_order, text, sha256) "
                "select id, 'other', 99, 'x', 'x' from documents limit 1"
            )
        )


def test_downgrade_and_upgrade_again(database_url: str) -> None:
    downgrade(database_url)
    assert "chunks" not in inspect(make_engine(database_url)).get_table_names()
    migrate(database_url)
    assert "chunks" in inspect(make_engine(database_url)).get_table_names()
