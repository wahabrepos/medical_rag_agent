"""BGE-small sentence embeddings with ONNX Runtime on CPU.

Matches the research-work embedding exactly in method: the [CLS] token of the last
hidden state, L2-normalised, with inputs truncated to 512 tokens and no query
instruction prefix. Uses the ONNX export published in the model repository.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
# Pinned so re-embedding the corpus later gives the same vectors.
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_DIM = 384
MAX_LENGTH = 512


class BgeEmbedder:
    model_id = BGE_MODEL_ID
    revision = BGE_REVISION
    dim = BGE_DIM

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        token = os.environ.get("HF_TOKEN") or None
        onnx_path = hf_hub_download(
            BGE_MODEL_ID, "onnx/model.onnx", revision=BGE_REVISION, cache_dir=cache_dir, token=token
        )
        tok_path = hf_hub_download(
            BGE_MODEL_ID, "tokenizer.json", revision=BGE_REVISION, cache_dir=cache_dir, token=token
        )
        # Separate tokenizer without truncation for counting (chunking needs true lengths).
        self._counter = Tokenizer.from_file(tok_path)
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=max_length)

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    @property
    def version(self) -> str:
        return f"{BGE_MODEL_ID}@{BGE_REVISION[:12]}+onnx-cls"

    def count_tokens(self, text: str) -> int:
        """Tokens including [CLS] and [SEP], without truncation."""
        return len(self._counter.encode(text).ids)

    def embed(self, texts: Sequence[str], *, batch_size: int = 8) -> npt.NDArray[np.float32]:
        """Normalised CLS embeddings, shape (len(texts), 384).

        Texts are batched in order of token length so each batch needs little padding;
        results are returned in input order.
        """
        out = np.empty((len(texts), BGE_DIM), dtype=np.float32)
        encodings = self.tokenizer.encode_batch(list(texts))
        order = sorted(range(len(texts)), key=lambda i: len(encodings[i].ids))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = len(encodings[i].ids)
                ids[row, :n] = encodings[i].ids
                mask[row, :n] = 1
            feeds = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            hidden = self.session.run(None, feeds)[0]
            cls = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            out[idx] = cls / np.maximum(norms, 1e-12)
        return out

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self.embed([text])[0]]
