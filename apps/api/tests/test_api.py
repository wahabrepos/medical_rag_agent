import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
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
from medrag_core.evidence import AnswerPolicy
from medrag_core.policy import FinalAnswerRule, LoopSettings
from medrag_core.verification import Verification
from medrag_search.knowledge import KnowledgePassage

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
    support_threshold = 0.7

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
    def __init__(
        self,
        answer_json: str,
        fail: Exception | None = None,
        passages: list[Passage] | None = None,
    ) -> None:
        self.nli = FakeNli()
        self.answer_json = answer_json
        self.fail = fail
        self.passages = passages if passages is not None else list(PASSAGES)

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
            retrieve=lambda q: list(self.passages),
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
        policy: AnswerPolicy = AnswerPolicy.EVIDENCE_GATED,
    ) -> TestClient:
        service = AnswerService(
            components=components,
            repository=InMemoryRunRepository(),
            ledger=SpendLedger(tmp_path / "spend.json", cap=cap),
            policy=policy,
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
    client = make_client(FakeComponents(answer("B", "supported claim")))

    body = client.post("/v1/ask", json={"question": "Which drug?"}, headers=AUTH).json()

    assert body["answer"] == body["model_answer"] == "B"
    assert body["note"] is None
    assert body["evidence"]["status"] == "supported"
    (statement,) = body["evidence"]["statements"]
    assert (statement["supported"], statement["supporting_pmid"]) == (True, 111)
    assert body["citations"][statement["supporting_citation"]]["pmid"] == 111
    assert [c["pmid"] for c in body["citations"]] == [111, 222]
    assert body["citations"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert body["cost"] > 0
    assert "not medical advice" in body["disclaimer"]


def test_partly_supported_answers_are_withheld(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "supported claim", "other claim")))

    body = client.post("/v1/ask", json={"question": "Which drug?"}, headers=AUTH).json()

    assert body["evidence"]["status"] == "partially_supported"
    assert (body["answer"], body["model_answer"]) == ("insufficient evidence", None)
    assert "do not support every part" in body["note"]
    assert "B" not in body["note"]
    # the ungrounded statement would reveal the withheld answer: only its count is returned
    (grounded,) = body["evidence"]["statements"]
    assert grounded["supported"]
    assert body["evidence"]["hidden_statements"] == 1
    assert [c["pmid"] for c in body["citations"]] == [111, 222]  # related studies still listed


def test_unverified_answer_only_on_request(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("no", "other claim")))

    body = client.post(
        "/v1/ask",
        json={"question": "Does X help?", "answer_format": "yes_no", "include_unverified": True},
        headers=AUTH,
    ).json()

    assert (body["answer"], body["model_answer"]) == ("insufficient evidence", "no")
    assert 'Unverified model answer (not backed by these studies): "no"' in body["note"]
    assert body["evidence"]["hidden_statements"] == 0
    assert [s["text"] for s in body["evidence"]["statements"]] == ["other claim"]


def test_uncertain_yes_no_policy(make_client: Callable[..., TestClient]) -> None:
    client = make_client(
        FakeComponents(answer("no", "other claim")), policy=AnswerPolicy.UNCERTAIN_YES_NO
    )

    body = client.post(
        "/v1/ask", json={"question": "Does X help?", "answer_format": "yes_no"}, headers=AUTH
    ).json()

    assert body["evidence"]["status"] == "not_supported"
    assert (body["answer"], body["model_answer"]) == ("uncertain", None)


def test_contradicted_answers_are_flagged(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("A", "contradicted claim")))
    ask = {"question": "Which?", "include_unverified": True}  # withheld: statements on request
    body = client.post("/v1/ask", json=ask, headers=AUTH).json()

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


def stream_data(client: TestClient, body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    with client.stream("POST", "/v1/ask/stream", json=body, headers=AUTH) as r:
        lines = list(r.iter_lines())
    events = {}
    for name, data in pairwise(lines):
        if name.startswith("event: ") and data.startswith("data: "):
            events[name.removeprefix("event: ")] = json.loads(data.removeprefix("data: "))
    return events


def test_gated_stream_does_not_leak_draft_answers(make_client: Callable[..., TestClient]) -> None:
    client = make_client(FakeComponents(answer("B", "other claim")))

    gated = stream_data(client, {"question": "Which drug?"})
    unverified = stream_data(client, {"question": "Which drug?", "include_unverified": True})

    assert "draft_answer" not in gated["generated"]
    assert gated["answer"]["answer"] == "insufficient evidence"
    assert unverified["generated"]["draft_answer"] == "B"


def test_cors_only_for_listed_origins() -> None:
    app = create_app(lambda: None, cors_origins=["https://ui.example"])  # type: ignore[arg-type,return-value]
    with TestClient(app) as client:
        allowed = client.options(
            "/v1/ask",
            headers={"Origin": "https://ui.example", "Access-Control-Request-Method": "POST"},
        )
        other = client.options(
            "/v1/ask",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert allowed.headers["access-control-allow-origin"] == "https://ui.example"
    assert "access-control-allow-origin" not in other.headers


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
        .post(
            "/v1/ask", json={"question": "What treats X?", "include_unverified": True}, headers=AUTH
        )
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


def test_textbook_citations_have_no_pubmed_link(make_client: Callable[..., TestClient]) -> None:
    book = KnowledgePassage(
        -(10**10), -2, 0, "InternalMed_Harrison: Aspirin", PASSAGES[0].text, book="Harrison"
    )
    client = make_client(FakeComponents(answer("B", "supported claim"), passages=[book]))

    body = client.post("/v1/ask", json={"question": "Which drug?"}, headers=AUTH).json()

    assert body["answer"] == "B"
    (citation,) = body["citations"]
    assert (citation["source"], citation["pmid"], citation["url"]) == ("textbook", None, None)
    assert body["evidence"]["statements"][0]["supporting_pmid"] is None


def test_medlineplus_citations_link_to_the_topic(make_client: Callable[..., TestClient]) -> None:
    topic = KnowledgePassage(
        -(10**10),
        -2,
        0,
        "Diabetes",
        PASSAGES[0].text,
        book="MedlinePlus",
        source="medlineplus",
        url="https://medlineplus.gov/diabetes.html",
    )
    client = make_client(FakeComponents(answer("B", "supported claim"), passages=[topic]))

    body = client.post("/v1/ask", json={"question": "Which drug?"}, headers=AUTH).json()

    (citation,) = body["citations"]
    assert (citation["source"], citation["pmid"], citation["url"]) == (
        "medlineplus",
        None,
        "https://medlineplus.gov/diabetes.html",
    )


def test_rewritten_query_is_streamed(make_client: Callable[..., TestClient]) -> None:
    components = FakeComponents(answer("B", "supported claim"))
    original = components.new_graph

    def with_rewrite(generator: LlmGenerator) -> Any:
        graph = original(generator)
        return (
            build_graph(
                retrieve=lambda q: list(PASSAGES),
                generate=generator,
                verify=lambda r, c: Verification(
                    support_score=0.9, unsupported=[], best_scores=[0.9]
                ),
                settings=LoopSettings(final_answer_rule=FinalAnswerRule.BEST_SUPPORTED),
                rewrite=lambda q: "clinical terms",
            )
            if graph
            else graph
        )

    components.new_graph = with_rewrite  # type: ignore[method-assign]
    events = stream_data(make_client(components), {"question": "Which drug?"})

    assert events["rewritten"] == {"query": "clinical terms"}
