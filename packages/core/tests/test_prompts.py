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
