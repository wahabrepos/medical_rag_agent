"""Evaluation sets: the research-work manifest, the golden subset and the full sets."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrag_eval.metrics import Dataset

EVAL_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = EVAL_ROOT / "reference"
GOLDEN_PATH = EVAL_ROOT / "golden" / "golden_150.jsonl"

OPTION_LABELS = "ABCDE"
MEDQA_INSTRUCTION = "Output ONLY the single letter (A/B/C/D) of the correct answer."


def format_medqa_question(question: str, options: list[str]) -> str:
    """MedQA question with lettered answer choices, as the research work sent it."""
    if not options:
        return question
    choices = "\n".join(
        f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(options[: len(OPTION_LABELS)])
    )
    return f"{question}\n\nAnswer choices:\n{choices}\n\n{MEDQA_INSTRUCTION}"


@dataclass(frozen=True)
class EvalItem:
    id: str
    dataset: Dataset
    question: str
    answer: str
    options: list[str] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)

    @property
    def binary_answer(self) -> bool:
        """PubMedQA questions get the yes/no constraint in the prompt."""
        return self.dataset is Dataset.PUBMEDQA


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_manifest() -> dict[Dataset, list[dict[str, Any]]]:
    data = json.loads((REFERENCE_DIR / "eval_sets_manifest.json").read_text("utf-8"))
    return {ds: data[ds.value] for ds in Dataset}


def load_golden(path: Path = GOLDEN_PATH) -> list[EvalItem]:
    return [
        EvalItem(
            id=row["id"],
            dataset=Dataset(row["dataset"]),
            question=row["question"],
            answer=row["answer"],
            options=row.get("options") or [],
            reference=row.get("reference", {}),
        )
        for row in _read_jsonl(path)
    ]


def load_full_set(dataset: Dataset, data_dir: Path) -> list[EvalItem]:
    """A full set written by eval/scripts/fetch_eval_sets.py (data/eval/<dataset>.jsonl)."""
    return [
        EvalItem(
            id=row["id"],
            dataset=dataset,
            question=row["question"],
            answer=row["answer"],
            options=row.get("options") or [],
        )
        for row in _read_jsonl(data_dir / f"{dataset.value}.jsonl")
    ]


def load_reference_predictions(system: str) -> list[dict[str, Any]]:
    return list(_read_jsonl(REFERENCE_DIR / "predictions" / f"{system}.jsonl"))
