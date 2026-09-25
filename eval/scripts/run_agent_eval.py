"""Run the Self-MedRAG agent over an evaluation set, resumably, and report metrics.

    uv run --env-file .env python eval/scripts/run_agent_eval.py --set golden --run v1-golden
    uv run --env-file .env python eval/scripts/run_agent_eval.py --set full --run v1-full

Each answered question is appended to eval/runs/<run>/predictions.jsonl straight
away; running the same command again skips questions already answered. When the
LLM provider's daily quota is exhausted the run stops with exit code 3 and can be
resumed later. The report compares accuracy and loop statistics with the research
work's Self-MedRAG + Mistral-small predictions on the same questions.

Heavy on memory (embedder + NLI): on the Jetson, run it under a memory cap, e.g.
    systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 uv run ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from medrag_eval.datasets import EvalItem, load_full_set, load_golden, load_reference_predictions
from medrag_eval.metrics import Dataset, evaluate

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SYSTEM = "selfmedrag_mistral"
EXIT_QUOTA = 3


def load_items(which: str, data_dir: Path) -> list[EvalItem]:
    if which == "golden":
        return load_golden()
    return [item for ds in Dataset for item in load_full_set(ds, data_dir)]


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def report(run_dir: Path, items: list[EvalItem]) -> dict[str, Any]:
    predictions = read_predictions(run_dir / "predictions.jsonl")
    reference = {row["id"]: row for row in load_reference_predictions(REFERENCE_SYSTEM)}
    out: dict[str, Any] = {"run": run_dir.name, "datasets": {}}
    for ds in Dataset:
        done = [i for i in items if i.dataset is ds and i.id in predictions]
        if not done:
            continue
        truth = [i.answer for i in done]
        new = evaluate([predictions[i.id] for i in done], truth, ds)
        old = evaluate([reference[i.id] for i in done], truth, ds)
        stops: dict[str, int] = {}
        for i in done:
            reason = predictions[i.id]["stop_reason"]
            stops[reason] = stops.get(reason, 0) + 1
        out["datasets"][ds.value] = {
            "questions": len(done),
            "of": sum(1 for i in items if i.dataset is ds),
            "agent": new,
            "research_work_same_questions": old,
            "stop_reasons": stops,
            "llm_calls": sum(predictions[i.id]["llm_calls"] for i in done),
            # Research-work quirk: an empty answer has rationale [""], which NLI scores as
            # "neutral", so it can be accepted. Counted so it stays visible.
            "empty_answers": sum(1 for i in done if not predictions[i.id]["final_answer"].strip()),
        }
    (run_dir / "report.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def print_report(result: dict[str, Any]) -> None:
    for name, ds in result["datasets"].items():
        a, r = ds["agent"], ds["research_work_same_questions"]
        print(
            f"{name:9} {ds['questions']}/{ds['of']}  accuracy {a['accuracy']:.4f} "
            f"(research work {r['accuracy']:.4f})  avg iterations {a['avg_iterations']:.3f} "
            f"(rw {r['avg_iterations']:.3f})  >1 iteration {a['more_than_one_iter_pct']:.1f}%  "
            f"avg support {a['avg_support_score']:.3f}  empty {ds['empty_answers']}  "
            f"stops {ds['stop_reasons']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=("golden", "full"), default="golden")
    ap.add_argument("--run", required=True, help="name of the run directory under eval/runs/")
    ap.add_argument(
        "--limit", type=int, default=None, help="answer at most this many new questions"
    )
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data/indexes")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    items = load_items(args.set, args.data_dir)
    run_dir = ROOT / "eval/runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.jsonl"

    if not args.report_only:
        from medrag_agent.llm import QuotaExhaustedError
        from medrag_agent.runtime import build_parity_agent
        from medrag_settings import get_settings

        done = read_predictions(out_path)
        todo = [i for i in items if i.id not in done][: args.limit]
        print(f"{len(done)} already answered, {len(todo)} to go", flush=True)
        agent = build_parity_agent(get_settings(), index_dir=args.index_dir)
        with out_path.open("a", encoding="utf-8") as fh:
            for n, item in enumerate(todo, 1):
                calls_before = agent.generator.calls
                started = time.monotonic()
                try:
                    state = agent.run(item.question, binary_answer=item.binary_answer)
                except QuotaExhaustedError as exc:
                    print(f"LLM quota exhausted after {n - 1} questions: {exc}", flush=True)
                    print_report(report(run_dir, items))
                    return EXIT_QUOTA
                history = state.get("history", [])
                row = {
                    "id": item.id,
                    "dataset": item.dataset.value,
                    "final_answer": state["answer"],
                    "rationale": state["rationale"],
                    "iterations": state["iteration"],
                    "support_score": state["support_score"],
                    "confidence": history[-1].confidence if history else 0.0,
                    "stop_reason": state["stop_reason"],
                    "latency": round(time.monotonic() - started, 3),
                    "llm_calls": agent.generator.calls - calls_before,
                    "model": agent.generator.config.model,
                    "history": [
                        {"query": h.query, "answer": h.answer, "support_score": h.support_score}
                        for h in history
                    ],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(
                    f"[{n}/{len(todo)}] {item.id} -> {row['final_answer'][:40]!r} "
                    f"(gold {item.answer}, it {row['iterations']}, {row['latency']:.0f}s)",
                    flush=True,
                )

    print_report(report(run_dir, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
