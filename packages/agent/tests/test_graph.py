import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from medrag_agent.errors import QuotaExhaustedError
from medrag_agent.graph import build_graph, to_loop_result
from medrag_core.loop import Generation, IterationRecord, StopReason, run_self_medrag
from medrag_core.policy import FinalAnswerRule, LoopSettings, RefinementStrategy
from medrag_core.verification import Verification

CORE_FIXTURES = Path(__file__).parents[2] / "core" / "tests" / "fixtures" / "research_work"
FIXTURE = json.loads((CORE_FIXTURES / "stop_rule_cases.json").read_text("utf-8"))
QUERY = "  What is the question?  "


class Fakes:
    """The fakes the research-work Trainer was captured with (see eval/capture)."""

    def __init__(self, script: list[Any], top_k: int = 5, refusals: set[int] | None = None) -> None:
        self.script = script
        self.refusals = refusals or set()
        self.top_k = top_k
        self.queries: list[str] = []
        self.generations = 0
        self.verifications = 0

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        return list(dict.fromkeys(["b1", "b2", "d1", "b1"]))[: self.top_k]

    def generate(
        self,
        query: str,
        context: list[str],
        history: list[IterationRecord],
        *,
        binary_answer: bool = False,
        multiple_choice: bool = False,
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


def run_graph(fakes: Fakes, settings: LoopSettings, clock: Any = None) -> Any:
    kwargs = {"clock": clock} if clock else {}
    graph = build_graph(
        retrieve=fakes.retrieve,
        generate=fakes.generate,
        verify=fakes.verify,
        settings=settings,
        **kwargs,
    )
    return to_loop_result(graph.invoke({"question": QUERY}))


def settings_for(strategy: str) -> LoopSettings:
    s = FIXTURE["settings"]
    return LoopSettings(
        max_iterations=s["max_iterations"],
        early_stopping=s["early_stopping"],
        min_improvement=s["min_improvement"],
        rationale_score_threshold=s["rationale_score_threshold"],
        refinement_strategy=RefinementStrategy(strategy),
    )


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_graph_matches_research_work(case: dict[str, Any]) -> None:
    fakes = Fakes(case["reflector_script"], FIXTURE["settings"]["top_k"])

    result = run_graph(fakes, settings_for(case["refinement_strategy"]))

    expected = case["expected"]
    assert result.answer == expected["answer"]
    assert result.rationale == expected["rationale"]
    assert result.iterations == expected["iterations"]
    assert result.support_score == expected["support_score"]
    assert fakes.queries == expected["retrieval_queries"]
    history = [{k: v for k, v in asdict(r).items() if k != "context"} for r in result.history]
    assert history == expected["history"]


@pytest.mark.parametrize("seed", range(300))
def test_graph_equals_core_loop(seed: int) -> None:
    rng = random.Random(seed)
    script: list[Any] = []
    for _ in range(3):
        if rng.random() < 0.1:
            script.append("raise")
        else:
            unsupported = [f"claim {i}" for i in range(rng.randint(0, 4))]
            script.append((round(rng.random(), 2), unsupported))
    settings = LoopSettings(
        early_stopping=rng.random() < 0.8,
        refinement_strategy=rng.choice(list(RefinementStrategy)),
        final_answer_rule=rng.choice(list(FinalAnswerRule)),
        prefer_committed_answers=rng.random() < 0.5,
    )
    refusals = {n for n in (1, 2, 3) if rng.random() < 0.3}

    graph_fakes, loop_fakes = Fakes(script, refusals=refusals), Fakes(script, refusals=refusals)
    from_graph = run_graph(graph_fakes, settings)
    from_loop = run_self_medrag(
        QUERY,
        retrieve=loop_fakes.retrieve,
        generate=loop_fakes.generate,
        verify=loop_fakes.verify,
        settings=settings,
    )

    assert from_graph == from_loop
    assert graph_fakes.queries == loop_fakes.queries


def test_timeout_stops_before_retrieving() -> None:
    ticks = iter([0.0, 1000.0])
    fakes = Fakes([(0.5, [])])

    result = run_graph(fakes, LoopSettings(), clock=lambda: next(ticks))

    assert result.stop_reason is StopReason.TIMEOUT
    assert result.answer == ""
    assert fakes.queries == []


def test_answer_format_flags_reach_generator() -> None:
    seen: list[bool] = []
    fakes = Fakes([(0.9, [])])

    def generate(
        *args: Any, binary_answer: bool = False, multiple_choice: bool = False
    ) -> Generation:
        seen.append((binary_answer, multiple_choice))
        return fakes.generate(*args)

    graph = build_graph(retrieve=fakes.retrieve, generate=generate, verify=fakes.verify)
    graph.invoke({"question": "Is it?", "binary_answer": True})
    graph.invoke({"question": "Which?", "multiple_choice": True})

    assert seen == [(True, False), (False, True)]


@pytest.mark.parametrize("stage", ["retrieve", "generate", "verify"])
def test_provider_errors_stop_the_run_instead_of_becoming_answers(stage: str) -> None:
    fakes = Fakes([(0.1, ["x"]), (0.2, ["x"]), (0.9, [])])

    def quota(*args: Any, **kwargs: Any) -> Any:
        raise QuotaExhaustedError("daily tokens used up")

    parts = {"retrieve": fakes.retrieve, "generate": fakes.generate, "verify": fakes.verify}
    parts[stage] = quota
    graph = build_graph(**parts)

    with pytest.raises(QuotaExhaustedError):
        graph.invoke({"question": QUERY})
