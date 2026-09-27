"""Export real evaluation answers as API responses for the web UI's mock mode.

The web UI can run without the API (NEXT_PUBLIC_API_URL=mock) by replaying these.
Each fixture is built by the API's own response code from a recorded golden-set
answer: the answer's passages are re-retrieved (retrieval is deterministic) and the
verifier is replaced by the scores the run recorded, so statuses, quotes and
"from the question" flags are exactly what the product showed. No LLM or GPU is used.

    uv run --env-file .env python eval/scripts/export_ui_fixtures.py \
        --out apps/web/src/mocks/fixtures.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from analyze_grounding import final_iteration

from medrag_api.schemas import AskRequest
from medrag_api.service import AnswerService
from medrag_core.evidence import (
    AnswerFormat,
    AnswerPolicy,
    QuotedClaim,
    normalize_statement,
    question_facts,
    quote_in_passage,
)
from medrag_eval.datasets import load_golden

ROOT = Path(__file__).resolve().parents[2]
# (run, question id, label shown in the UI's example list)
CHOSEN = [
    ("v3e-quotes", "pubmedqa-labeled-788", "Answer shown: every statement quoted and verified"),
    (
        "v3g-quotes-textbooks",
        "medqa-test-568",
        "Withheld: a finding restated from the question, but no source for the reasoning",
    ),
    ("v3e-quotes", "pubmedqa-labeled-173", "Withheld: the only quote states the study's aim"),
    ("v3e-quotes", "medqa-test-1189", "Withheld: clinical reasoning not found in the literature"),
    (
        "v3g-quotes-textbooks",
        "medqa-test-862",
        "Textbook source: every statement verified, yet the wrong option",
    ),
]


class RecordedVerifier:
    """Returns the support the run recorded for each statement's quoted passage,
    and for statements grounded in the question (recorded with the current rules by
    apply_question_grounding.py); 0 for everything else."""

    support_threshold = 0.5
    version = "recorded scores (lytang/MiniCheck-RoBERTa-Large)"

    def __init__(self, row: dict[str, Any], question: str) -> None:
        self.facts = question_facts(question)
        self.by_claim = {}
        flags = row.get("from_question") or [False] * len(row["statements"])
        for statement, from_question in zip(row["statements"], flags, strict=True):
            self.by_claim[normalize_statement(statement["text"])] = (statement, from_question)

    def probabilities(self, pairs: Sequence[tuple[str, str]], **_: Any) -> list[list[float]]:
        out = []
        for passage, claim in pairs:
            statement, from_question = self.by_claim.get(claim, (None, False))
            p = 0.0
            if statement is not None:
                if passage == self.facts:
                    p = 0.95 if from_question else 0.0
                elif statement["quote"] and quote_in_passage(statement["quote"], passage):
                    p = statement["support"]
            out.append([0.0, p, 1.0 - p])
        return out


def progress_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for i, h in enumerate(row["history"], start=1):
        events.append({"event": "retrieved", "data": {"iteration": i, "passages": 5}})
        events.append({"event": "generated", "data": {"iteration": i}})
        events.append(
            {"event": "verified", "data": {"iteration": i, "support": h["support_score"]}}
        )
        if i < len(row["history"]):
            events.append({"event": "refining", "data": {"next_iteration": i + 1}})
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data/indexes")
    args = ap.parse_args()

    from medrag_agent.runtime import build_components
    from medrag_settings import get_settings

    items = {i.id: i for i in load_golden()}
    retrievers: dict[str | None, Any] = {}
    fixtures = []
    for run, qid, label in CHOSEN:
        run_dir = ROOT / "eval/runs" / run
        source = run_dir / "predictions.question-grounded.jsonl"
        rows = {r["id"]: r for r in map(json.loads, source.read_text("utf-8").splitlines()) if r}
        row = rows[qid]
        knowledge = row.get("knowledge_dir")
        if knowledge not in retrievers:
            retrievers[knowledge] = build_components(
                get_settings(),
                index_dir=args.index_dir,
                support_label="entailment",
                knowledge_dir=ROOT / knowledge if knowledge else None,
            ).retriever
        item = items[qid]
        index = final_iteration(row)
        passages = retrievers[knowledge].passages(row["history"][index]["query"])
        quotes = tuple(QuotedClaim(q["claim"], q["passage"], q["quote"]) for q in row["quotes"])
        state = {
            "question": item.question,
            "answer": row["final_answer"],
            "rationale": row["rationale"],
            "final_iteration": 1,
            "iteration_passages": [passages],
            "iteration_claims": [None],
            "iteration_evidence": [quotes],
            "iteration": row["iterations"],
            "stop_reason": row["stop_reason"],
        }
        fmt = AnswerFormat.MULTIPLE_CHOICE if item.dataset.value == "medqa" else AnswerFormat.YES_NO
        components = SimpleNamespace(nli=RecordedVerifier(row, item.question), evidence_quotes=True)
        service = AnswerService(components, None, None, policy=AnswerPolicy.EVIDENCE_GATED)  # type: ignore[arg-type]
        request = AskRequest(question=item.question, answer_format=fmt, include_unverified=True)
        response = service._response(
            request, state, row["model"], float(row["latency"]), float(row["cost"])
        )
        response.run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"medrag-fixture/{run}/{qid}")
        payload = response.model_dump(mode="json")
        if payload["answer"] != (
            row["final_answer"] if row["gated_shown"] else "insufficient evidence"
        ):
            raise SystemExit(f"{qid}: rebuilt gate decision differs from the recorded run")
        fixtures.append(
            {
                "id": qid,
                "label": label,
                "source_run": run,
                "gold_answer": item.answer,
                "request": {"question": item.question, "answer_format": fmt.value},
                "events": progress_events(row),
                "answer": payload,
            }
        )
        print(f"{qid}: {payload['answer']!r} ({payload['evidence']['status']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixtures, indent=1, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
