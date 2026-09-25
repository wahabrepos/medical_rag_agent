from pathlib import Path

import pytest

from medrag_agent.budget import BudgetExceededError, SpendLedger, cost


def test_cost_uses_price_table() -> None:
    # 1M input tokens at 0.15 + 1M output tokens at 0.75
    assert cost("groq/openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.90)


def test_unknown_model_is_refused() -> None:
    with pytest.raises(BudgetExceededError, match="no price known"):
        cost("someone/unknown-model", 10, 10)


def test_ledger_accumulates_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    SpendLedger(path).add(
        run="a", model="groq/openai/gpt-oss-120b", prompt_tokens=1_000_000, completion_tokens=0
    )
    SpendLedger(path).add(
        run="b", model="mistral/mistral-small-2603", prompt_tokens=0, completion_tokens=1_000_000
    )
    assert SpendLedger(path).total == pytest.approx(0.45)


def test_check_stops_before_passing_ninety_percent(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", cap=1.0)
    ledger.add(
        run="a", model="groq/openai/gpt-oss-120b", prompt_tokens=0, completion_tokens=1_100_000
    )

    ledger.check(0.05)  # 0.825 + 0.05 <= 0.90
    with pytest.raises(BudgetExceededError):
        ledger.check(0.10)
