"""Run the Self-MedRAG agent over an evaluation set, resumably, and report metrics.

    uv run --env-file .env python eval/scripts/run_agent_eval.py --set golden --run v1-golden
    uv run --env-file .env python eval/scripts/run_agent_eval.py --set full --run v1-full

Each answered question is appended to eval/runs/<run>/predictions.jsonl straight
away; running the same command again skips questions already answered. When the
LLM provider's daily quota is exhausted the run stops with exit code 3 and can be
resumed later; answers recorded as errors are dropped on resume and asked again.
LLM spend is recorded in data/llm_spend.json for all runs; a run stops (exit code 4)
before a question that could take total spend past 90% of --budget.
The report compares accuracy and loop statistics with the research
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
from medrag_eval.metrics import Dataset, evaluate, grounded_metrics

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SYSTEM = "selfmedrag_mistral"
EXIT_QUOTA = 3
EXIT_BUDGET = 4
LEDGER = ROOT / "data/llm_spend.json"


def load_items(which: str, data_dir: Path) -> list[EvalItem]:
    if which == "golden":
        return load_golden()
    return [item for ds in Dataset for item in load_full_set(ds, data_dir)]


def load_source_pmids(data_dir: Path) -> dict[str, int]:
    """PubMedQA question id -> PubMed id of its source article (from fetch_eval_sets.py)."""
    path = data_dir / "pubmedqa.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing: run eval/scripts/fetch_eval_sets.py first")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return {row["id"]: int(row["pubid"]) for row in rows if row.get("pubid")}


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def drop_errors(path: Path) -> dict[str, dict[str, Any]]:
    """Remove answers recorded as errors so they are asked again; returns the rest."""
    rows = read_predictions(path)
    kept = {k: v for k, v in rows.items() if v["stop_reason"] != "error"}
    if len(kept) < len(rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept.values()))
        print(f"dropped {len(rows) - len(kept)} error answers; they will be asked again")
    return kept


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
        support_labels = {predictions[i.id].get("support_label", "neutral") for i in done}
        out["datasets"][ds.value] = {
            "questions": len(done),
            "support_label": ",".join(sorted(support_labels)),
            # Meaningful only with the entailment verifier; see medrag_eval.metrics.
            "grounded": grounded_metrics([predictions[i.id] for i in done], truth, ds),
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
        g = ds["grounded"]
        grounded_acc = g.get("grounded_accuracy")
        print(
            f"{'':9} grounded ({ds['support_label']} verifier): "
            f"{g['grounded_share']:.1%} of answers"
            + (f", accuracy {grounded_acc:.4f}" if grounded_acc is not None else "")
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
    ap.add_argument(
        "--support-label",
        choices=("neutral", "entailment"),
        default="neutral",
        help='NLI column counted as support: "neutral" reproduces the research work, '
        '"entailment" is the corrected verifier',
    )
    ap.add_argument(
        "--final-answer-rule",
        choices=("research_work", "best_supported"),
        default="research_work",
        help="answer returned when the loop stalls or runs out of iterations",
    )
    ap.add_argument(
        "--prefer-committed",
        action="store_true",
        help='never return "insufficient evidence" when an earlier iteration answered',
    )
    ap.add_argument(
        "--mcq-commit",
        action="store_true",
        help="MedQA prompts require an option letter instead of allowing a refusal",
    )
    ap.add_argument(
        "--lenient-json",
        action="store_true",
        help="read answer fields from invalid JSON before the research-work fallback",
    )
    ap.add_argument(
        "--nli-url",
        default=None,
        help="use a remote inference service for NLI, e.g. http://localhost:8001 over a tunnel",
    )
    ap.add_argument(
        "--normalize-statements",
        action="store_true",
        help='verifier checks the claim of "Passage N states that X" statements (X)',
    )
    ap.add_argument(
        "--leakage-free",
        action="store_true",
        help="PubMedQA: never retrieve the question's own source article",
    )
    ap.add_argument(
        "--only",
        choices=("medqa", "pubmedqa"),
        default=None,
        help="evaluate only one dataset",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=3.0,
        help="total LLM spend cap across all runs and providers (stops at 90%%)",
    )
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    items = load_items(args.set, args.data_dir)
    if args.only:
        items = [i for i in items if i.dataset.value == args.only]
    source_pmid = load_source_pmids(args.data_dir) if args.leakage_free else {}
    run_dir = ROOT / "eval/runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.jsonl"

    if not args.report_only:
        from medrag_agent.budget import BudgetExceededError, SpendLedger, cost
        from medrag_agent.errors import ProviderUnavailableError
        from medrag_agent.runtime import build_parity_agent
        from medrag_core.policy import FinalAnswerRule, LoopSettings
        from medrag_settings import get_settings

        done = drop_errors(out_path)
        todo = [i for i in items if i.id not in done][: args.limit]
        print(f"{len(done)} already answered, {len(todo)} to go", flush=True)
        loop = LoopSettings(
            final_answer_rule=FinalAnswerRule(args.final_answer_rule),
            prefer_committed_answers=args.prefer_committed,
        )
        agent = build_parity_agent(
            get_settings(),
            index_dir=args.index_dir,
            support_label=args.support_label,
            loop=loop,
            lenient_json=args.lenient_json,
            nli_url=args.nli_url,
            normalize_statements=args.normalize_statements,
        )
        ledger = SpendLedger(LEDGER, cap=args.budget)
        gen = agent.generator
        model = gen.config.model
        # Worst case for one question: 3 iterations of a long prompt and a full completion.
        worst_case = cost(model, 3 * 3_000, 3 * gen.config.max_tokens)
        print(f"LLM spend so far {ledger.total:.4f} of {ledger.limit:.2f} allowed", flush=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for n, item in enumerate(todo, 1):
                try:
                    ledger.check(worst_case)
                except BudgetExceededError as exc:
                    print(f"Stopping: {exc}", flush=True)
                    print_report(report(run_dir, items))
                    return EXIT_BUDGET
                calls_before = gen.calls
                raw_before = len(gen.raw_outputs)
                tokens_before = (gen.prompt_tokens, gen.completion_tokens)
                started = time.monotonic()
                try:
                    state = agent.run(
                        item.question,
                        binary_answer=item.binary_answer,
                        multiple_choice=args.mcq_commit and item.dataset is Dataset.MEDQA,
                        exclude_pmids=[source_pmid[item.id]] if item.id in source_pmid else None,
                    )
                except ProviderUnavailableError as exc:
                    print(f"LLM quota exhausted after {n - 1} questions: {exc}", flush=True)
                    print_report(report(run_dir, items))
                    return EXIT_QUOTA
                prompt_tokens = gen.prompt_tokens - tokens_before[0]
                completion_tokens = gen.completion_tokens - tokens_before[1]
                spent = ledger.add(
                    run=args.run,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    note=item.id,
                )
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
                    "llm_calls": gen.calls - calls_before,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": round(spent, 6),
                    "model": agent.generator.config.model,
                    "support_label": args.support_label,
                    "final_answer_rule": args.final_answer_rule,
                    "prefer_committed": args.prefer_committed,
                    "mcq_commit": args.mcq_commit,
                    "lenient_json": args.lenient_json,
                    "normalize_statements": args.normalize_statements,
                    "excluded_pmid": source_pmid.get(item.id),
                    "nli": getattr(agent.nli, "version", "local"),
                    "raw_outputs": gen.raw_outputs[raw_before:],
                    "history": [
                        {"query": h.query, "answer": h.answer, "support_score": h.support_score}
                        for h in history
                    ],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(
                    f"[{n}/{len(todo)}] {item.id} -> {row['final_answer'][:40]!r} "
                    f"(gold {item.answer}, it {row['iterations']}, {row['latency']:.0f}s, "
                    f"spend {ledger.total:.4f})",
                    flush=True,
                )

    print_report(report(run_dir, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
