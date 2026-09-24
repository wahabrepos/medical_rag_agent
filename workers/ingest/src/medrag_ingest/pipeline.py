"""Ingestion: corpus -> documents and sections -> chunks -> embeddings -> indexes.

Each stage can be run on its own. Embedding only processes chunks that have no
embedding yet, so an interrupted run can be resumed.
"""

import hashlib
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sqlalchemy import Engine, func, insert, select, text, update
from sqlalchemy.orm import Session

from medrag_db.indexes import PROFILES, create_hnsw_index_sql, drop_hnsw_index_sql
from medrag_db.models import Chunk, Document, Section
from medrag_ingest.chunking import chunk_by_tokens, chunk_research_work
from medrag_ingest.corpus import SectionRecord
from medrag_search.bm25 import BM25Index

logger = logging.getLogger(__name__)

STANDARD_MAX_TOKENS = 350


class Embedder(Protocol):
    @property
    def version(self) -> str: ...

    model_id: str

    def count_tokens(self, text: str) -> int: ...

    def embed(self, texts: Sequence[str], *, batch_size: int = ...) -> npt.NDArray[np.float32]: ...


@dataclass(frozen=True)
class LoadReport:
    documents: int
    sections: int
    chunks: dict[str, int]


def sha256(text_: str) -> str:
    return hashlib.sha256(text_.encode("utf-8")).hexdigest()


def reset_corpus(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents, sections, chunks RESTART IDENTITY CASCADE"))


def load_corpus(
    engine: Engine,
    records: Iterable[SectionRecord],
    *,
    profiles: Sequence[str] = PROFILES,
    count_tokens: Callable[[str], int],
    standard_max_tokens: int = STANDARD_MAX_TOKENS,
) -> LoadReport:
    """Insert documents, sections and chunks (without embeddings) into empty tables."""
    records = list(records)
    with Session(engine) as session, session.begin():
        doc_rows: dict[int, dict[str, object]] = {}
        for r in records:
            doc_rows.setdefault(
                r.pmid, {"pmid": r.pmid, "source": r.subset, "title": r.title, "meta": r.meta}
            )
        doc_ids = {
            pmid: doc_id
            for doc_id, pmid in session.execute(
                insert(Document).returning(Document.id, Document.pmid), list(doc_rows.values())
            )
        }
        section_ids = {
            pos: sid
            for sid, pos in session.execute(
                insert(Section).returning(Section.id, Section.corpus_position),
                [
                    {
                        "document_id": doc_ids[r.pmid],
                        "position": r.section_position,
                        "label": r.label,
                        "text": r.text,
                        "corpus_position": r.corpus_position,
                    }
                    for r in records
                ],
            )
        }

        counts: dict[str, int] = {}
        if "parity" in profiles:
            rows: list[dict[str, object]] = []
            for r in records:
                for piece in chunk_research_work(r.text):
                    rows.append(
                        {
                            "document_id": doc_ids[r.pmid],
                            "section_id": section_ids[r.corpus_position],
                            "profile": "parity",
                            "corpus_order": len(rows),
                            "text": piece,
                            "sha256": sha256(piece),
                            "n_tokens": count_tokens(piece),
                        }
                    )
            session.execute(insert(Chunk), rows)
            counts["parity"] = len(rows)

        if "standard" in profiles:
            by_doc: dict[int, list[SectionRecord]] = {}
            for r in records:
                by_doc.setdefault(r.pmid, []).append(r)
            rows = []
            for pmid, sections in by_doc.items():
                body = "\n".join(f"{s.label}: {s.text}" if s.label else s.text for s in sections)
                for piece in chunk_by_tokens(body, count_tokens, max_tokens=standard_max_tokens):
                    rows.append(
                        {
                            "document_id": doc_ids[pmid],
                            "section_id": None,
                            "profile": "standard",
                            "corpus_order": len(rows),
                            "text": piece,
                            "sha256": sha256(piece),
                            "n_tokens": count_tokens(piece),
                        }
                    )
            session.execute(insert(Chunk), rows)
            counts["standard"] = len(rows)

    report = LoadReport(len(doc_ids), len(section_ids), counts)
    logger.info("loaded %s", report)
    return report


def embed_missing(
    engine: Engine, embedder: Embedder, *, profile: str, page_size: int = 512, batch_size: int = 8
) -> int:
    """Embed every chunk of `profile` that has no embedding yet; returns how many."""
    done = 0
    started = time.monotonic()
    with Session(engine) as session:
        total = session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.profile == profile, Chunk.embedding.is_(None))
        )
    while True:
        with Session(engine) as session, session.begin():
            page = session.execute(
                select(Chunk.id, Chunk.text)
                .where(Chunk.profile == profile, Chunk.embedding.is_(None))
                .order_by(Chunk.corpus_order)
                .limit(page_size)
            ).all()
            if not page:
                break
            vectors = embedder.embed([row.text for row in page], batch_size=batch_size)
            session.execute(
                update(Chunk),
                [
                    {
                        "id": row.id,
                        "embedding": vec,
                        "embed_model": embedder.model_id,
                        "embed_version": embedder.version,
                    }
                    for row, vec in zip(page, vectors, strict=True)
                ],
            )
        done += len(page)
        rate = done / max(time.monotonic() - started, 1e-9)
        logger.info("%s: embedded %d/%s (%.1f chunks/s)", profile, done, total, rate)
    return done


def build_vector_index(engine: Engine, *, profile: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(drop_hnsw_index_sql(profile)))
        conn.execute(text(create_hnsw_index_sql(profile)))
        conn.execute(text("ANALYZE chunks"))


def build_bm25(engine: Engine, *, profile: str, path: Path) -> BM25Index:
    """BM25 over the profile's chunks in corpus order (the order the research work used)."""
    with Session(engine) as session:
        rows = session.execute(
            select(Chunk.id, Chunk.text)
            .where(Chunk.profile == profile)
            .order_by(Chunk.corpus_order)
        ).all()
    index = BM25Index.from_texts([r.text for r in rows], [r.id for r in rows])
    index.save(path)
    logger.info("%s: BM25 over %d chunks -> %s", profile, index.size, path)
    return index
