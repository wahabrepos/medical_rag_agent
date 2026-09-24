"""Initial schema: documents, sections, chunks (pgvector), runs, feedback.

Revision ID: 0001
Revises:
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

from medrag_db.indexes import PROFILES, create_hnsw_index_sql, drop_hnsw_index_sql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("pmid", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "sections",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("label", sa.Text()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("corpus_position", sa.Integer(), nullable=False, unique=True),
        sa.UniqueConstraint("document_id", "position"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section_id", sa.BigInteger(), sa.ForeignKey("sections.id", ondelete="CASCADE")),
        sa.Column("profile", sa.String(16), nullable=False),
        sa.Column("corpus_order", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("n_tokens", sa.Integer()),
        sa.Column("embedding", HALFVEC(384)),
        sa.Column("embed_model", sa.String(128)),
        sa.Column("embed_version", sa.String(64)),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("profile", "corpus_order"),
        sa.CheckConstraint("profile in ('parity', 'standard')", name="chunk_profile"),
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index("ix_chunks_document", "chunks", ["document_id"])
    for profile in PROFILES:
        op.execute(create_hnsw_index_sql(profile))

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stop_reason", sa.String(32)),
        sa.Column("iterations", sa.SmallInteger()),
        sa.Column("support_score", sa.Float()),
        sa.Column("model", sa.String(128)),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("rating in (-1, 1)", name="feedback_rating"),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("runs")
    for profile in PROFILES:
        op.execute(drop_hnsw_index_sql(profile))
    op.drop_table("chunks")
    op.drop_table("sections")
    op.drop_table("documents")
