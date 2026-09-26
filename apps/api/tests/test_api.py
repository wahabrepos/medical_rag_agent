import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medrag_agent.budget import SpendLedger
from medrag_agent.errors import QuotaExhaustedError
from medrag_agent.graph import build_graph
from medrag_agent.llm import GeneratorConfig, LlmGenerator, RateLimiter
from medrag_api.app import ApiContext, create_app
from medrag_api.repository import InMemoryRunRepository
from medrag_api.service import AnswerService
from medrag_core.policy import FinalAnswerRule, LoopSettings
from medrag_core.verification import Verification

KEY = "test-key"


@dataclass(frozen=True)
class Passage:
    pmid: int
    title: str
    text: str


PASSAGES = [
    Passage(111, "Aspirin trial", "Aspirin reduced events."),
    Passage(222, "Other study", "Unrelated finding."),
]


class FakeNli:
    """Entailment 0.9 for (passage 0, 'supported claim'); contradiction for 'contradicted claim'."""

    version = "fake-nli"

    def probabilities(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        rows = []
        for passage, statement in pairs:
            if statement.lower() == "supported claim" and passage == PASSAGES[0].text:
                rows.append([0.0, 0.9, 0.1])
            elif statement.lower() == "contradicted claim" and passage == PASSAGES[1].text:
                rows.append([0.9, 0.0, 0.1])
            else:
                rows.append([0.0, 0.1, 0.9])
        return rows


class FakeComponents:
    def __init__(self, answer_json: str, fail: Exception | None = None) -> None:
        self.nli = FakeNli()
        self.answer_json = answer_json
        self.fail = fail

    def completion(self, **kwargs: Any) -> Any:
        if self.fail:
            raise self.fail
        return SimpleNamespace(
            model="groq/openai/gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer_json), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )

    def new_generator(self) -> LlmGenerator:
        return LlmGenerator(
            GeneratorConfig(),
            completion=self.completion,
            limiter=RateLimiter(1000, 10**9),
        )

    def new_graph(self, generator: LlmGenerator) -> Any:
        def verify(rationale: list[str], context: list[str]) -> Verification:
            return Verification(support_score=0.9, unsupported=[], best_scores=[0.9])

        return build_graph(
            retrieve=lambda q: list(PASSAGES),
            generate=generator,
            verify=verify,
            settings=LoopSettings(final_answer_rule=FinalAnswerRule.BEST_SUPPORTED),
        )


def answer(text: str, *rationale: str) -> str:
    return json.dumps({"answer": text, "rationale": list(rationale), "confidence": 0.8})


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(
        components: FakeComponents,
        *,
        cap: float = 3.0,
        open_access: bool = False,
        per_minute: int = 100,
    ) -> TestClient:
        service = AnswerService(
            components=components,
            repository=InMemoryRunRepository(),
            ledger=SpendLedger(tmp_path / "spend.json", cap=cap),
        )
        ctx = ApiContext(
            service, api_keys={KEY}, open_access=open_access, requests_per_minute=per_minute
        )
        client = TestClient(create_app(lambda: ctx))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


AUTH = {"Authorization": f"Bearer {KEY}"}


def test_answer_with_evidence_and_citations(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "supported claim", "other claim")))

    body = client.post("/v1/ask", json={"question": "Which drug?"}, headers=AUTH).json()

    assert body["answer"] == body["model_answer"] == "B"
    assert body["evidence"]["status"] == "partially_supported"
    first, second = body["evidence"]["statements"]
    assert (first["supported"], first["supporting_pmid"]) == (True, 111)
    assert (second["supported"], second["supporting_pmid"]) == (False, None)
    assert [c["pmid"] for c in body["citations"]] == [111, 222]
    assert body["citations"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert body["cost"] > 0
    assert "not medical advice" in body["disclaimer"]


def test_unsupported_yes_no_becomes_uncertain(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("no", "other claim")))

    body = client.post(
        "/v1/ask", json={"question": "Does X help?", "answer_format": "yes_no"}, headers=AUTH
    ).json()

    assert body["evidence"]["status"] == "not_supported"
    assert (body["answer"], body["model_answer"]) == ("uncertain", "no")
    assert "leans" in body["note"]


def test_contradicted_answers_are_flagged(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("A", "contradicted claim")))
    body = client.post("/v1/ask", json={"question": "Which?"}, headers=AUTH).json()

    assert body["evidence"]["status"] == "contradicted"
    assert body["evidence"]["statements"][0]["contradicting_pmid"] == 222


def test_stream_sends_progress_then_answer(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "supported claim")))

    with client.stream(
        "POST", "/v1/ask/stream", json={"question": "What treats X?"}, headers=AUTH
    ) as r:
        events = [
            line.removeprefix("event: ") for line in r.iter_lines() if line.startswith("event:")
        ]

    assert events == ["retrieved", "generated", "verified", "answer"]


def test_runs_are_stored_and_take_feedback(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "supported claim")))
    run_id = client.post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH).json()[
        "run_id"
    ]

    assert client.get(f"/v1/runs/{run_id}", headers=AUTH).json()["answer"] == "B"
    ok = client.post(f"/v1/runs/{run_id}/feedback", json={"rating": 1}, headers=AUTH)
    bad = client.post(f"/v1/runs/{run_id}/feedback", json={"rating": 5}, headers=AUTH)
    missing = client.post(
        "/v1/runs/00000000-0000-0000-0000-000000000000/feedback", json={"rating": 1}, headers=AUTH
    )
    assert (ok.status_code, bad.status_code, missing.status_code) == (201, 422, 404)


def test_authentication_and_rate_limit(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "x")), per_minute=1)
    assert client.post("/v1/ask", json={"question": "What treats X?"}).status_code == 401
    assert (
        client.post(
            "/v1/ask", json={"question": "What treats X?"}, headers={"Authorization": "Bearer x"}
        ).status_code
        == 401
    )
    assert (
        client.post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH).status_code == 200
    )
    assert (
        client.post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH).status_code == 429
    )


def test_local_open_access(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "x")), open_access=True)
    assert client.post("/v1/ask", json={"question": "What treats X?"}).status_code == 200


def test_budget_and_provider_outages_return_503(make_client: Callable[..., TestClient]) -> None:
    broke = make_client(FakeComponents(answer("B", "x")), cap=0.0001)
    down = make_client(FakeComponents(answer("B", "x"), fail=QuotaExhaustedError("quota")))

    assert (
        broke.post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH).status_code == 503
    )
    assert (
        down.post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH).status_code == 503
    )


def test_health_and_validation(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "x")))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    assert client.post("/v1/ask", json={"question": "x"}, headers=AUTH).status_code == 422


def test_answer_claim_is_assessed_first(make_client: Callable[..., TestClient]) -> None:
    text = json.dumps({"answer": "A", "rationale": ["other claim"], "claim": "contradicted claim"})
    components = FakeComponents(text)
    original = components.new_generator

    def with_claim() -> LlmGenerator:
        generator = original()
        generator.config = GeneratorConfig(answer_claim=True)
        return generator

    components.new_generator = with_claim  # type: ignore[method-assign]
    body = (
        make_client(components)
        .post("/v1/ask", json={"question": "What treats X?"}, headers=AUTH)
        .json()
    )

    first = body["evidence"]["statements"][0]
    assert (first["kind"], first["text"], first["contradicted"]) == (
        "claim",
        "contradicted claim",
        True,
    )
    assert body["evidence"]["status"] == "contradicted"
    assert body["evidence"]["statements"][1]["kind"] == "rationale"
