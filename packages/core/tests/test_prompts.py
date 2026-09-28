import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from medrag_core.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_messages, build_prompt

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("prompt_cases.json")


@dataclass(frozen=True)
class _Entry:
    query: str
    answer: str
    support_score: float


def _history(case: dict[str, Any]) -> list[_Entry]:
    return [_Entry(h["query"], h["answer"], h["support_score"]) for h in case["history"]]


def test_prompt_constants_match_research_work() -> None:
    assert FIXTURE["system_prompt"] == SYSTEM_PROMPT
    assert FIXTURE["user_prompt_template"] == USER_PROMPT_TEMPLATE


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_prompt_matches_research_work(case: dict[str, Any]) -> None:
    binary = case["dataset_type"] == "pubmedqa"

    prompt = build_prompt(case["query"], case["context"], _history(case), binary_answer=binary)
    messages = build_messages(case["query"], case["context"], _history(case), binary_answer=binary)

    assert prompt == case["prompt"]
    assert messages.system == case["messages"]["system"]
    assert messages.user == case["messages"]["user"]


def test_multiple_choice_instruction_is_optional_and_last() -> None:
    plain = build_messages("Q?", ["p"], [])
    mcq = build_messages("Q?", ["p"], [], multiple_choice=True)

    assert "multiple-choice" not in plain.user
    assert mcq.user.startswith(plain.user)
    assert mcq.user.endswith('say so in the "rationale" field.')
    assert mcq.system == plain.system


def test_answer_claim_instruction_is_optional() -> None:
    plain = build_messages("Q?", ["p"], [])
    with_claim = build_messages("Q?", ["p"], [], answer_claim=True)

    assert '"claim"' not in plain.user
    assert with_claim.user.startswith(plain.user)
    assert '"claim" field' in with_claim.user


def test_evidence_quotes_instruction_is_optional() -> None:
    plain = build_messages("Q?", ["p"], [])
    quoted = build_messages("Q?", ["p"], [], evidence_quotes=True)

    assert '"evidence"' not in plain.user
    assert quoted.user.startswith(plain.user)
    assert '"evidence" list' in quoted.user
    assert "word for word" in quoted.user


def test_answer_claim_needs_a_quote_when_both_options_are_on() -> None:
    both = build_messages("Q?", ["p"], [], answer_claim=True, evidence_quotes=True)
    quotes_only = build_messages("Q?", ["p"], [], evidence_quotes=True)

    assert both.user.endswith("when a passage supports it.")
    assert '"claim" field needs an evidence entry' not in quotes_only.user


def test_rewrite_prompt_and_parsing() -> None:
    from medrag_core.prompts import build_rewrite_messages, parse_rewrite

    messages = build_rewrite_messages("  I'm always thirsty  ")
    assert messages.user == "Question: I'm always thirsty"
    assert "clinical terms" in messages.system

    assert parse_rewrite('"polyuria polydipsia"\nbecause ...', "q") == "polyuria polydipsia"
    assert parse_rewrite("Query: polyuria", "q") == "polyuria"
    assert parse_rewrite('polyuria AND (diabetes OR "thirst")', "q") == "polyuria diabetes thirst"
    assert parse_rewrite("   \n", " the question ") == "the question"
    assert len(parse_rewrite(" ".join(["word"] * 100), "q").split()) == 40
