from collections.abc import Iterator

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from medrag_inference.app import Models, create_app
from medrag_inference.client import InferenceUnavailableError, RemoteNli

from .test_app import FakeEmbedder, FakeNli


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> Iterator[RemoteNli]:
    app = create_app(lambda: Models(embedder=FakeEmbedder(), nli=FakeNli()))
    with TestClient(app) as test_client:
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: test_client)
        yield RemoteNli("http://inference")


def test_remote_probabilities_and_scorer(remote: RemoteNli) -> None:
    pairs = [("p1", "s1"), ("p2", "s2"), ("p3", "s3")]

    probs = remote.probabilities(pairs, batch_size=2)  # two requests

    assert remote.version == "fake-nli@1"
    assert probs.shape == (3, 3)
    assert np.allclose(probs[:, 1], 0.2)
    assert remote.scorer("neutral")(pairs) == pytest.approx([0.7, 0.7, 0.7])


def test_unreachable_service_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("tunnel down", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(refuse), **kwargs),
    )
    with pytest.raises(InferenceUnavailableError, match="tunnel down"):
        RemoteNli("http://localhost:1", retries=2)
