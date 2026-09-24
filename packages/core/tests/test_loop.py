import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from medrag_core.loop import (
    Generation,
    IterationRecord,
    StopReason,
    run_self_medrag,
)
from medrag_core.policy import LoopSettings, RefinementStrategy
from medrag_core.verification import Verification

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("stop_rule_cases.json")
QUERY = "  What is the question?  "


def _settings(strategy: str) -> LoopSettings:
    s = FIXTURE["settings"]
    return LoopSettings(
        max_iterations=s["max_iterations"],
        early_stopping=s["early_stopping"],
        min_improvement=s["min_improvement"],
        rationale_score_threshold=s["rationale_score_threshold"],
        refinement_strategy=RefinementStrategy(strategy),
    )


class _Fakes:
    """Same fakes the research-work Trainer was run with in eval/capture."""

    def __init__(self, script: list[Any], top_k: int) -> None:
        self.script = script
        self.top_k = top_k
        self.queries: list[str] = []
        self.generations = 0
        self.verifications = 0

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        return list(dict.fromkeys(["b1", "b2", "d1", "b1"]))[: self.top_k]

    def generate(
        self, query: str, context: list[str], history: list[IterationRecord]
    ) -> Generation:
        self.generations += 1
        n = self.generations
        return Generation(
            f"answer-{n}", [f"stmt-{n}-a", f"stmt-{n}-b"], 0.5 + n / 10, [f"cite-{n}"]
        )

    def verify(self, rationale: list[str], context: list[str]) -> Verification:
        step = self.script[self.verifications]
        self.verifications += 1
        if step == "raise":
            raise RuntimeError("scripted failure")
        score, unsupported = step
        return Verification(support_score=score, unsupported=list(unsupported), best_scores=[])


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_loop_matches_research_work(case: dict[str, Any]) -> None:
    fakes = _Fakes(case["reflector_script"], FIXTURE["settings"]["top_k"])

    result = run_self_medrag(
        QUERY,
        retrieve=fakes.retrieve,
        generate=fakes.generate,
        verify=fakes.verify,
        settings=_settings(case["refinement_strategy"]),
    )

    expected = case["expected"]
    assert result.answer == expected["answer"]
    assert result.rationale == expected["rationale"]
    assert result.iterations == expected["iterations"]
    assert result.support_score == expected["support_score"]
    assert fakes.queries == expected["retrieval_queries"]
    history = [{k: v for k, v in asdict(r).items() if k != "context"} for r in result.history]
    assert history == expected["history"]


def test_stop_reasons() -> None:
    def run(script: list[Any]) -> StopReason:
        f = _Fakes(script, 5)
        return run_self_medrag(
            QUERY, retrieve=f.retrieve, generate=f.generate, verify=f.verify
        ).stop_reason

    assert run([(0.8, [])]) is StopReason.ACCEPTED
    assert run([(0.5, ["x"]), (0.52, ["x"])]) is StopReason.STALLED
    assert run([(0.4, ["x"]), (0.5, ["x"]), (0.6, ["x"])]) is StopReason.MAX_ITERATIONS
    assert run(["raise"]) is StopReason.ERROR


def test_timeout_returns_empty_answer() -> None:
    ticks = iter([0.0, 1000.0])
    f = _Fakes([(0.5, [])], 5)

    result = run_self_medrag(
        QUERY,
        retrieve=f.retrieve,
        generate=f.generate,
        verify=f.verify,
        clock=lambda: next(ticks),
    )

    assert result.stop_reason is StopReason.TIMEOUT
    assert result.answer == ""
    assert result.iterations == 0


def test_reported_confidence_is_from_last_iteration() -> None:
    f = _Fakes([(0.4, ["x"]), (0.6, ["x"]), (0.5, ["x"])], 5)
    settings = LoopSettings(early_stopping=False)

    result = run_self_medrag(
        QUERY, retrieve=f.retrieve, generate=f.generate, verify=f.verify, settings=settings
    )

    assert result.answer == "answer-2"  # best-of-history
    assert result.confidence == pytest.approx(0.8)  # third iteration, as in the research work
