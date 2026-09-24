"""PostgreSQL + pgvector schema and sessions."""

from medrag_db.models import EMBEDDING_DIM, Base, Chunk, Document, Feedback, Run, Section
from medrag_db.session import make_engine, make_session_factory

__all__ = [
    "EMBEDDING_DIM",
    "Base",
    "Chunk",
    "Document",
    "Feedback",
    "Run",
    "Section",
    "make_engine",
    "make_session_factory",
]
__version__ = "0.1.0"
