import json
from pathlib import Path
from typing import Any

import pytest

from medrag_ingest.chunking import chunk_by_tokens, chunk_research_work, split_sentences

FIXTURES = Path(__file__).parent / "fixtures" / "research_work"


def load_fixture(name: str) -> Any:
    """Expected behaviour captured from the research-work code (see eval/capture)."""
    return json.loads((FIXTURES / name).read_text("utf-8"))


FIXTURE = load_fixture("chunker_cases.json")


def words(text: str) -> int:
    """Stand-in token counter: one token per word."""
    return len(text.split())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_research_work_chunker_is_identical(case: dict[str, Any]) -> None:
    got = chunk_research_work(
        case["text"],
        chunk_size=FIXTURE["chunk_size_words"],
        overlap=FIXTURE["overlap"],
        min_words=FIXTURE["min_words"],
    )
    assert got == case["chunks"]


def test_sentence_splitter_keeps_abbreviations() -> None:
    text = "Dose was 5 mg. Results vary, e.g. In adults. Smith et al. Reported this. 3 cases."

    assert split_sentences(text) == [
        "Dose was 5 mg.",
        "Results vary, e.g. In adults.",
        "Smith et al. Reported this.",
        "3 cases.",
    ]


def test_decimal_numbers_do_not_split() -> None:
    assert split_sentences("The ratio was 1.5 in total. Next one.") == [
        "The ratio was 1.5 in total.",
        "Next one.",
    ]


def test_token_chunks_respect_limit_and_overlap() -> None:
    sentences = [f"Sentence {i} has five words." for i in range(20)]
    chunks = chunk_by_tokens(" ".join(sentences), words, max_tokens=12, overlap_sentences=1)

    assert all(words(c) <= 12 for c in chunks)
    assert chunks[0] == f"{sentences[0]} {sentences[1]}"
    assert chunks[1].startswith(sentences[1])  # one sentence of overlap
    assert chunks[-1].endswith(sentences[-1])


def test_long_sentence_is_split_into_windows() -> None:
    chunks = chunk_by_tokens(" ".join(["word"] * 25), words, max_tokens=10)

    assert [words(c) for c in chunks] == [10, 10, 5]


def test_short_text_is_one_chunk() -> None:
    assert chunk_by_tokens("Just one sentence.", words) == ["Just one sentence."]
    assert chunk_by_tokens("", words) == []
