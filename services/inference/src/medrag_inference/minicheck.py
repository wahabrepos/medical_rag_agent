"""Claim verifier: MiniCheck-RoBERTa-Large (lytang/MiniCheck-RoBERTa-Large, MIT).

A fact-checking model trained to decide whether a document supports a claim, used
in place of general-purpose NLI for grounding. On hand-labelled statement-passage
pairs from our runs it ranked support far better than nli-deberta-v3-base and did
not mark unrelated passages as supporting (eval/grounding/verifier_benchmark.json);
at its default threshold of 0.5 it held 100% precision on a held-out sample
(eval/grounding/validation_pairs.jsonl).

It has two classes (unsupported, supported) and no contradiction class. To keep the
NLI interface, probabilities are returned in LABELS order as
(contradiction = 0, entailment = P(supported), neutral = P(unsupported)), so the
support label is "entailment" and nothing is ever reported as contradicted.

The Hugging Face repo only has PyTorch weights: the ONNX file is exported once with
services/inference/scripts/export_minicheck.py (compared with PyTorch there) and
its path configured with VERIFIER_ONNX_PATH.
"""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download

from medrag_inference.nli import DEFAULT_CACHE_SIZE, MAX_LENGTH, OnnxPairClassifier

MINICHECK_MODEL_ID = "lytang/MiniCheck-RoBERTa-Large"
MINICHECK_REVISION = "74c8919647e61ed0f71bc177d94f10930f090068"
MINICHECK_THRESHOLD = 0.5


class MiniCheckVerifier(OnnxPairClassifier):
    model_id = MINICHECK_MODEL_ID
    revision = MINICHECK_REVISION
    support_threshold = MINICHECK_THRESHOLD

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        max_length: int = MAX_LENGTH,
        cache_size: int = DEFAULT_CACHE_SIZE,
        device: str = "cpu",
    ) -> None:
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(
                f"MiniCheck ONNX model not found at {onnx_path}; "
                "export it with services/inference/scripts/export_minicheck.py"
            )
        tok_path = hf_hub_download(
            MINICHECK_MODEL_ID,
            "tokenizer.json",
            revision=MINICHECK_REVISION,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
        # Like MiniCheck itself: only the passage is cut, the claim is kept whole.
        super().__init__(
            onnx_path,
            tok_path,
            truncation="only_first",
            threads=threads,
            max_length=max_length,
            cache_size=cache_size,
            device=device,
        )

    @property
    def version(self) -> str:
        return f"{MINICHECK_MODEL_ID}@{MINICHECK_REVISION[:12]}+onnx"

    def _to_labels(self, probs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        unsupported, supported = probs[:, 0], probs[:, 1]
        return np.stack([np.zeros_like(supported), supported, unsupported], axis=1)
