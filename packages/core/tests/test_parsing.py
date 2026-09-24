import json
from pathlib import Path
from typing import Any

import pytest

from medrag_core.parsing import ParsedGeneration, parse_generation_output

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


CASES = load_fixture("parser_cases.json")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_parser_matches_research_work(case: dict[str, Any]) -> None:
    assert parse_generation_output(case["input"]) == case["expected"]


def test_typed_view_of_valid_json() -> None:
    parsed = ParsedGeneration.from_text(
        '```json\n{"answer": "B", "rationale": "One step", "confidence": 0.9}\n```'
    )

    assert parsed.answer == "B"
    assert parsed.rationale == ["One step"]
    assert parsed.confidence == 0.9
    assert parsed.citations == []
    assert not parsed.used_fallback


def test_typed_view_flags_fallback_and_coerces_types() -> None:
    fallback = ParsedGeneration.from_text("Answer: C\nBecause the passage says so clearly.")
    odd = ParsedGeneration.from_text('{"answer": "A", "rationale": [1], "confidence": "high"}')

    assert fallback.used_fallback
    assert fallback.answer == "C"
    assert odd.rationale == ["1"]
    assert odd.confidence == 0.7
    assert not odd.used_fallback
