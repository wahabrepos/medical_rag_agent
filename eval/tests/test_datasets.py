import pytest

from medrag_eval.datasets import (
    MEDQA_INSTRUCTION,
    EvalItem,
    format_medqa_question,
    load_golden,
)
from medrag_eval.metrics import Dataset


def test_golden_loads_150_items() -> None:
    items = load_golden()

    assert len(items) == 150
    assert sum(i.dataset is Dataset.PUBMEDQA for i in items) == 75
    assert all(i.binary_answer == (i.dataset is Dataset.PUBMEDQA) for i in items)


MEDQA_ITEMS = [i for i in load_golden() if i.dataset is Dataset.MEDQA]


@pytest.mark.parametrize("item", MEDQA_ITEMS, ids=lambda i: i.id)
def test_medqa_formatting_reproduces_research_work(item: EvalItem) -> None:
    raw_question = item.question.split("\n\nAnswer choices:\n")[0]

    assert format_medqa_question(raw_question, item.options) == item.question
    assert item.question.endswith(MEDQA_INSTRUCTION)


def test_question_without_options_is_unchanged() -> None:
    assert format_medqa_question("Plain question?", []) == "Plain question?"
