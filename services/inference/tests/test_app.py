from collections.abc import Iterator

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
    assert body == {"status": "ok", "embedder": "fake-embedder@1", "nli": "fake-nli@1"}


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
