from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient

from medrag_inference.app import MAX_PAIRS, Models, create_app


class FakeEmbedder:
    version = "fake-embedder@1"

    def embed(self, texts: list[str], *, batch_size: int = 8) -> npt.NDArray[np.float32]:
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32)


class FakeNli:
    version = "fake-nli@1"
    support_threshold = 0.5

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def probabilities(
        self, pairs: list[tuple[str, str]], *, batch_size: int = 16
    ) -> npt.NDArray[np.float32]:
        self.seen.extend(pairs)
        return np.tile(np.array([0.1, 0.2, 0.7], dtype=np.float32), (len(pairs), 1))


@pytest.fixture
def nli() -> FakeNli:
    return FakeNli()


@pytest.fixture
def client(nli: FakeNli) -> Iterator[TestClient]:
    app = create_app(lambda: Models(embedder=FakeEmbedder(), nli=nli))
    with TestClient(app) as test_client:
        yield test_client


def test_healthz_reports_model_versions(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body == {
        "status": "ok",
        "embedder": "fake-embedder@1",
        "nli": "fake-nli@1",
        "support_threshold": 0.5,
    }


def test_embed(client: TestClient) -> None:
    body = client.post("/embed", json={"texts": ["abc", "hello"]}).json()
    assert body["model"] == "fake-embedder@1"
    assert body["dim"] == 3
    assert body["vectors"] == [[3.0, 1.0, 0.0], [5.0, 1.0, 0.0]]


def test_nli_returns_labelled_probabilities(client: TestClient, nli: FakeNli) -> None:
    body = client.post("/nli", json={"pairs": [["passage", "statement"]]}).json()
    assert body["labels"] == ["contradiction", "entailment", "neutral"]
    assert body["probabilities"] == [pytest.approx([0.1, 0.2, 0.7])]
    assert nli.seen == [("passage", "statement")]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/embed", {"texts": []}),
        ("/embed", {"texts": ["x" * 20_001]}),
        ("/nli", {"pairs": []}),
        ("/nli", {"pairs": [["only one"]]}),
        ("/nli", {"pairs": [["a", "b"]] * (MAX_PAIRS + 1)}),
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, path: str, payload: object) -> None:
    assert client.post(path, json=payload).status_code == 422


def test_nli_batches_respect_token_budget() -> None:
    from medrag_inference.nli import _batches

    lengths = [10] * 20 + [512] * 10
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = _batches(order, lengths, batch_size=16, max_tokens=4096)  # explicit budget

    assert sorted(i for b in batches for i in b) == list(range(30))
    assert all(len(b) <= 16 for b in batches)
    assert all(len(b) * max(lengths[i] for i in b) <= 4096 for b in batches)
    assert [len(b) for b in batches] == [16, 8, 6]  # 4 short + 4 long = 8 x 512 tokens


def test_minicheck_classes_map_onto_nli_labels() -> None:
    from medrag_inference.minicheck import MiniCheckVerifier

    verifier = MiniCheckVerifier.__new__(MiniCheckVerifier)  # mapping only, no model
    probs = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)

    labelled = verifier._to_labels(probs)

    # (contradiction, entailment, neutral) = (0, P(supported), P(unsupported))
    assert labelled.tolist() == [
        pytest.approx([0.0, 0.1, 0.9]),
        pytest.approx([0.0, 0.8, 0.2]),
    ]


def test_minicheck_without_exported_model_explains_how_to_get_it(tmp_path: Path) -> None:
    from medrag_inference.minicheck import MiniCheckVerifier

    with pytest.raises(FileNotFoundError, match=r"export_minicheck\.py"):
        MiniCheckVerifier(tmp_path / "missing.onnx")
