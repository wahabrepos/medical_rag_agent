"""CPU inference service: BGE embeddings and DeBERTa NLI."""

from medrag_inference.embedding import BGE_DIM, BGE_MODEL_ID, BGE_REVISION, BgeEmbedder
from medrag_inference.nli import (
    CORRECT_SUPPORT_LABEL,
    LABELS,
    RESEARCH_WORK_SUPPORT_LABEL,
    DebertaNli,
)

__all__ = [
    "BGE_DIM",
    "BGE_MODEL_ID",
    "BGE_REVISION",
    "CORRECT_SUPPORT_LABEL",
    "LABELS",
    "RESEARCH_WORK_SUPPORT_LABEL",
    "BgeEmbedder",
    "DebertaNli",
]
__version__ = "0.1.0"
