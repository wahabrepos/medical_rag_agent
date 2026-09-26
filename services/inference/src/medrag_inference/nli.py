"""NLI verifier: cross-encoder/nli-deberta-v3-base with ONNX Runtime on CPU.

Returns all three class probabilities per (premise, hypothesis) pair, in the
model's label order: 0 = contradiction, 1 = entailment, 2 = neutral.

Research-work note: the original code read column 2 as "entailment", but column 2
is *neutral*. The parity build keeps that behaviour (support label "neutral") so
results stay comparable; the corrected verifier uses "entailment".

The fp32 ONNX export is used. Every int8 variant (published exports and our own
dynamic quantisation, full or partial) failed the accuracy bar; see
services/inference/reports/nli_variants_x86_zen4.json. Scores of repeated
(premise, hypothesis) pairs are cached in memory.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
# Pinned so verification scores are reproducible.
NLI_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
LABELS = ("contradiction", "entailment", "neutral")
MAX_LENGTH = 512

# The label the research work (unknowingly) used as "entailment".
RESEARCH_WORK_SUPPORT_LABEL = "neutral"
CORRECT_SUPPORT_LABEL = "entailment"


MODEL_FILE = "onnx/model.onnx"
DEFAULT_CACHE_SIZE = 20_000


class DebertaNli:
    model_id = NLI_MODEL_ID
    revision = NLI_REVISION
    labels = LABELS

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        max_length: int = MAX_LENGTH,
        cache_size: int = DEFAULT_CACHE_SIZE,
        device: str = "cpu",
    ) -> None:
        token = os.environ.get("HF_TOKEN") or None
        self.model_file = MODEL_FILE
        model_path = hf_hub_download(
            NLI_MODEL_ID, self.model_file, revision=NLI_REVISION, cache_dir=cache_dir, token=token
        )
        tok_path = hf_hub_download(
            NLI_MODEL_ID, "tokenizer.json", revision=NLI_REVISION, cache_dir=cache_dir, token=token
        )
        self.tokenizer = Tokenizer.from_file(tok_path)
        # Same truncation as the research work: longest_first to 512 tokens.
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        self.tokenizer.no_padding()

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA requested but onnxruntime has no CUDA provider")
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._cache: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self._cache_size = cache_size
        # The HTTP service calls this from several threads.
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return f"{NLI_MODEL_ID}@{NLI_REVISION[:12]}+{self.model_file.rsplit('/', 1)[-1]}"

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 16
    ) -> npt.NDArray[np.float32]:
        """Softmax probabilities, shape (len(pairs), 3), columns in LABELS order.

        Each pair is (premise, hypothesis) = (retrieved passage, rationale statement).
        Pairs seen recently come from the cache; the rest are batched by length to
        keep padding small. Output is in input order.
        """
        out = np.empty((len(pairs), len(LABELS)), dtype=np.float32)
        keys = [_pair_key(p, h) for p, h in pairs]
        todo: list[int] = []
        with self._lock:
            for i, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is None:
                    todo.append(i)
                else:
                    self._cache.move_to_end(key)
                    out[i] = cached
        if todo:
            computed = self._compute([pairs[i] for i in todo], batch_size)
            with self._lock:
                for row, i in enumerate(todo):
                    out[i] = computed[row]
                    self._remember(keys[i], computed[row])
        return out

    def _remember(self, key: str, value: npt.NDArray[np.float32]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = value.copy()
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _compute(
        self, pairs: Sequence[tuple[str, str]], batch_size: int
    ) -> npt.NDArray[np.float32]:
        out = np.empty((len(pairs), len(LABELS)), dtype=np.float32)
        encodings = self.tokenizer.encode_batch([(p, h) for p, h in pairs])
        order = sorted(range(len(pairs)), key=lambda i: len(encodings[i].ids))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            types = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                enc = encodings[i]
                n = len(enc.ids)
                ids[row, :n] = enc.ids
                mask[row, :n] = 1
                types[row, :n] = enc.type_ids
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            logits = self.session.run(None, feeds)[0].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            out[idx] = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
        return out

    def scorer(self, label: str = CORRECT_SUPPORT_LABEL) -> "LabelScorer":
        """An NliScorer (see medrag_core.verification) returning one label's probability."""
        return LabelScorer(self, label)


def _pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{premise}\x00{hypothesis}".encode()).hexdigest()


class ProbabilityModel(Protocol):
    def probabilities(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = ...
    ) -> npt.NDArray[np.float32]: ...


class LabelScorer:
    """Callable adapter: pairs -> probability of one label per pair."""

    def __init__(self, nli: ProbabilityModel, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown NLI label {label!r}; expected one of {LABELS}")
        self.nli = nli
        self.label = label
        self.column = LABELS.index(label)

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(p) for p in self.nli.probabilities(pairs)[:, self.column]]
