from dataclasses import dataclass

import pytest

from medrag_core.loop import Generation, IterationRecord, StopReason, run_self_medrag
from medrag_core.policy import FinalAnswerRule, LoopSettings, choose_final, is_uncommitted
from medrag_core.verification import Verification


@dataclass(frozen=True)
class R:
    answer: str
    support_score: float


BEST = LoopSettings(final_answer_rule=FinalAnswerRule.BEST_SUPPORTED)
BEST_COMMITTED = LoopSettings(
    final_answer_rule=FinalAnswerRule.BEST_SUPPORTED, prefer_committed_answers=True
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("", True), ("  ", True), ("Insufficient evidence", True), ("B", False), ("no", False)],
)
def test_is_uncommitted(answer: str, expected: bool) -> None:
    assert is_uncommitted(answer) is expected


def test_research_work_rule_returns_latest_when_stalled() -> None:
    history = [R("A", 0.5), R("B", 0.1)]
    assert choose_final(history, stalled=True, settings=LoopSettings()).answer == "B"
    assert choose_final(history, stalled=False, settings=LoopSettings()).answer == "A"


def test_best_supported_keeps_the_earliest_on_ties() -> None:
    history = [R("A", 0.0), R("insufficient evidence", 0.0), R("C", 0.0)]
    assert choose_final(history, stalled=True, settings=BEST).answer == "A"


def test_committed_answer_wins_over_better_supported_refusal() -> None:
    history = [R("A", 0.1), R("Insufficient evidence", 0.5)]
    assert choose_final(history, stalled=True, settings=BEST).answer == "Insufficient evidence"
    assert choose_final(history, stalled=True, settings=BEST_COMMITTED).answer == "A"


def test_refusals_only_fall_back_to_best_refusal() -> None:
    history = [R("insufficient evidence", 0.1), R("", 0.3)]
    assert choose_final(history, stalled=False, settings=BEST_COMMITTED).support_score == 0.3


def test_loop_keeps_first_answer_when_second_backs_off() -> None:
    """The v2a failure: support stays 0, the second answer refuses, the loop stalls."""
    answers = iter(["C", "insufficient evidence"])

    def generate(query: str, context: list[str], history: list[IterationRecord]) -> Generation:
        return Generation(next(answers), ["claim"], 0.9, [])

    def run(settings: LoopSettings) -> str:
        return run_self_medrag(
            "Q",
            retrieve=lambda q: ["passage"],
            generate=generate,
            verify=lambda r, c: Verification(0.0, list(r), [0.0]),
            settings=settings,
        ).answer

    result = run_self_medrag(
        "Q",
        retrieve=lambda q: ["passage"],
        generate=lambda q, c, h: Generation(
            "C" if not h else "insufficient evidence", ["x"], 0.9, []
        ),
        verify=lambda r, c: Verification(0.0, list(r), [0.0]),
        settings=BEST_COMMITTED,
    )
    assert result.stop_reason is StopReason.STALLED
    assert result.answer == "C"
    assert run(LoopSettings()) == "insufficient evidence"  # research-work rule
