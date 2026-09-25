"""LLM spend tracking with a hard cap shared by all runs and providers.

Spend is estimated from each call's token counts and a per-model price table.
Prices are conservative estimates (USD per million tokens) to be checked against
the provider dashboards; unknown models are refused. Amounts are compared with
the cap one-to-one (1 USD counted as 1 EUR), which overstates EUR spend while the
euro is worth more than the dollar.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# (input, output) USD per 1M tokens. Conservative estimates; verify on the pricing pages.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "groq/openai/gpt-oss-120b": (0.15, 0.75),
    "mistral/mistral-small-2603": (0.10, 0.30),
    "mistral/mistral-small-latest": (0.10, 0.30),
}


class BudgetExceededError(RuntimeError):
    """The next request could take total LLM spend past the cap."""


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICES_PER_MTOK:
        raise BudgetExceededError(f"no price known for {model!r}; refusing to spend on it")
    price_in, price_out = PRICES_PER_MTOK[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


@dataclass
class SpendLedger:
    path: Path
    cap: float = 3.0
    stop_at_fraction: float = 0.9  # keep a margin: prices are estimates

    @property
    def limit(self) -> float:
        return self.cap * self.stop_at_fraction

    def _load(self) -> dict[str, object]:
        if self.path.exists():
            data: dict[str, object] = json.loads(self.path.read_text("utf-8"))
            return data
        return {"total": 0.0, "entries": []}

    @property
    def total(self) -> float:
        return float(self._load()["total"])  # type: ignore[arg-type]

    def check(self, next_estimate: float) -> None:
        if self.total + next_estimate > self.limit:
            raise BudgetExceededError(
                f"LLM spend {self.total:.4f} + next {next_estimate:.4f} would pass "
                f"{self.limit:.2f} (90% of the {self.cap:.2f} cap)"
            )

    def add(
        self, *, run: str, model: str, prompt_tokens: int, completion_tokens: int, note: str = ""
    ) -> float:
        amount = cost(model, prompt_tokens, completion_tokens)
        data = self._load()
        entries = data["entries"]
        if not isinstance(entries, list):
            raise ValueError(f"corrupt spend ledger: {self.path}")
        entries.append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "run": run,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": round(amount, 6),
                "note": note,
            }
        )
        data["total"] = round(float(data["total"]) + amount, 6)  # type: ignore[arg-type]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n")
        return amount
