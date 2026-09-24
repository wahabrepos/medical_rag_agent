"""CPU inference service: BGE embeddings and DeBERTa NLI."""

from medrag_inference.embedding import BGE_DIM, BGE_MODEL_ID, BGE_REVISION, BgeEmbedder

__all__ = ["BGE_DIM", "BGE_MODEL_ID", "BGE_REVISION", "BgeEmbedder"]
__version__ = "0.1.0"
