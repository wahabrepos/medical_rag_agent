"""Dense retrieval over pgvector (HNSW, cosine distance on halfvec embeddings)."""

from collections.abc import Collection, Sequence

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from medrag_db.indexes import HNSW_EF_SEARCH
from medrag_db.models import Chunk


def dense_search(
    session: Session,
    embedding: Sequence[float],
    *,
    profile: str,
    k: int = 10,
    ef_search: int = HNSW_EF_SEARCH,
    exclude_document_ids: Collection[int] = (),
) -> list[int]:
    """Chunk ids of the k nearest chunks of one profile, nearest first."""
    session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    stmt = select(Chunk.id).where(Chunk.profile == profile)
    if exclude_document_ids:
        stmt = stmt.where(Chunk.document_id.not_in(list(exclude_document_ids)))
    stmt = stmt.order_by(Chunk.embedding.cosine_distance(list(embedding))).limit(k)
    return list(session.scalars(stmt))
