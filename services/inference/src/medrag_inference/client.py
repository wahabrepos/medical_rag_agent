"""HTTP client for a remote inference service (see medrag_inference.app).

Lets the agent run NLI on another machine, for example a rented GPU reached
through an SSH tunnel, while retrieval and LLM calls stay local.
"""

from collections.abc import Sequence

import httpx
import numpy as np
import numpy.typing as npt

from medrag_inference.app import MAX_PAIRS
from medrag_inference.nli import LABELS, LabelScorer


class InferenceUnavailableError(RuntimeError):
    """The remote inference service could not be reached or failed."""


class RemoteNli:
    def __init__(self, base_url: str, *, timeout: float = 120.0, retries: int = 3) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._retries = retries
        health = self._request("GET", "/healthz")
        self.version = str(health["nli"])

    def _request(self, method: str, path: str, json: object | None = None) -> dict[str, object]:
        last: Exception | None = None
        for _ in range(self._retries):
            try:
                response = self._client.request(method, path, json=json)
                response.raise_for_status()
                body: dict[str, object] = response.json()
                return body
            except httpx.HTTPError as exc:
                last = exc
        raise InferenceUnavailableError(f"inference service {method} {path} failed: {last}")

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = MAX_PAIRS
    ) -> npt.NDArray[np.float32]:
        out = np.empty((len(pairs), len(LABELS)), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
            chunk = [list(p) for p in pairs[start : start + batch_size]]
            body = self._request("POST", "/nli", {"pairs": chunk})
            if body["labels"] != list(LABELS):
                raise InferenceUnavailableError(f"unexpected label order {body['labels']}")
            probabilities = body["probabilities"]
            if not isinstance(probabilities, list):
                raise InferenceUnavailableError("malformed /nli response")
            out[start : start + len(chunk)] = np.asarray(probabilities, dtype=np.float32)
        return out

    def scorer(self, label: str) -> LabelScorer:
        return LabelScorer(self, label)
