"""MiniCheck ONNX against PyTorch reference scores (needs the exported model; -m model).

The fixture holds 24 statement-passage pairs from our runs, spread over the score
range, with P(supported) from PyTorch + transformers at the pinned revision. The
model file comes from services/inference/scripts/export_minicheck.py; the test is
skipped where it has not been exported.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from medrag_inference.minicheck import (
    MINICHECK_MODEL_ID,
    MINICHECK_REVISION,
    MINICHECK_THRESHOLD,
    MiniCheckVerifier,
)
from medrag_inference.nli import LABELS

pytestmark = pytest.mark.model

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "minicheck_cases.json").read_text())
ONNX_PATH = Path(os.environ.get("VERIFIER_ONNX_PATH", "data/models/minicheck-roberta-large.onnx"))
DEVICE = os.environ.get("MEDRAG_TEST_DEVICE", "cpu")


@pytest.fixture(scope="module")
def verifier() -> MiniCheckVerifier:
    if not ONNX_PATH.is_file():
        pytest.skip(f"MiniCheck ONNX not exported at {ONNX_PATH}")
    return MiniCheckVerifier(ONNX_PATH, cache_size=0, device=DEVICE)


def test_fixture_matches_pinned_model() -> None:
    assert (FIXTURE["model"], FIXTURE["revision"]) == (MINICHECK_MODEL_ID, MINICHECK_REVISION)


def test_support_matches_pytorch(verifier: MiniCheckVerifier) -> None:
    pairs = [(c["passage"], c["claim"]) for c in FIXTURE["cases"]]
    reference = np.array([c["supported"] for c in FIXTURE["cases"]])

    probs = verifier.probabilities(pairs)
    supported = probs[:, LABELS.index("entailment")]

    # Measured max difference: 0.0007 on CPU, 0.001 on CUDA (reports/minicheck_onnx_export.json).
    assert np.abs(supported - reference).max() < (2e-3 if DEVICE == "cpu" else 0.02)
    assert np.array_equal(supported >= MINICHECK_THRESHOLD, reference >= MINICHECK_THRESHOLD)
    assert np.all(probs[:, LABELS.index("contradiction")] == 0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_batching_does_not_change_scores(verifier: MiniCheckVerifier) -> None:
    pairs = [(c["passage"], c["claim"]) for c in FIXTURE["cases"]]

    together = verifier.probabilities(pairs)
    one_by_one = np.vstack([verifier.probabilities([p]) for p in pairs])

    # On CUDA, padding changes which kernels run: up to 5e-4 measured on an RTX 3060.
    assert np.abs(together - one_by_one).max() < (1e-4 if DEVICE == "cpu" else 2e-3)
