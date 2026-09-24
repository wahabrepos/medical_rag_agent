"""Database schema: PubMed documents, their abstract sections, retrieval chunks, runs.

A document is one PubMed article (PMID). A section is one labelled part of its
abstract, and is also one record of the research-work corpus. Chunks are the
retrieval units, built per chunking profile:

- "parity": the research-work chunking (one or more chunks per section), used by
  the parity build so retrieval matches the research work
- "standard": token-aware chunks over the whole abstract
"""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    pmid: Mapped[int] = mapped_column(BigInteger, unique=True)
    source: Mapped[str] = mapped_column(String(32))
    # PubMedQA stores the article's question-style title; kept as metadata, never
    # put into chunk text (it would leak evaluation questions into retrieval).
    title: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sections: Mapped[list["Section"]] = relationship(
        back_populates="document", order_by="Section.position"
    )


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("document_id", "position"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(SmallInteger)
    # Free-text PubMed section heading; some are long (e.g. structured-abstract labels).
    label: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    # 0-based record number in the research-work corpus order.
    corpus_position: Mapped[int] = mapped_column(Integer, unique=True)

    document: Mapped[Document] = relationship(back_populates="sections")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("profile", "corpus_order"),
        CheckConstraint("profile in ('parity', 'standard')", name="chunk_profile"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    profile: Mapped[str] = mapped_column(String(16))
    # Position in the profile's corpus order; BM25 is built in this order.
    corpus_order: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    n_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[Any] = mapped_column(HALFVEC(EMBEDDING_DIM), nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(128))
    embed_version: Mapped[str | None] = mapped_column(String(64))
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship()


class Run(Base):
    """One answered question (filled in by the API in a later step)."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    stop_reason: Mapped[str | None] = mapped_column(String(32))
    iterations: Mapped[int | None] = mapped_column(SmallInteger)
    support_score: Mapped[float | None] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint("rating in (-1, 1)", name="feedback_rating"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(SmallInteger)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
