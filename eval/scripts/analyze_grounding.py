"""How far can the agent's answers be trusted when only evidence-backed ones are shown?

Re-scores a finished run without new LLM calls: for every answer it retrieves the
passages of the iteration whose answer was returned (retrieval is deterministic),
scores every rationale statement against them with NLI and writes

- eval/runs/<run>/grounding/statements.jsonl: per answer, the statements with their
  best entailment / contradiction over the passages, and whether the answer is correct
- eval/runs/<run>/grounding/summary.json: accuracy and share per evidence status, and
  risk-coverage tables for several answer-level evidence scores
- eval/runs/<run>/grounding/label_sample.jsonl: statement-passage pairs sampled across
  entailment and contradiction levels, to be judged by hand (verifier precision)

Runs with live PubMed are not reproducible (search results change) and are refused.

    uv run --env-file .env python eval/scripts/analyze_grounding.py --run v3a-normalized \
        --set golden --nli-url http://localhost:18001 [--out-name grounding-minicheck]

The verifier (and its support threshold) is whatever the inference service serves.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from medrag_core.evidence import (
    CONTRADICTION,
    ENTAILMENT,
    EVIDENCE_THRESHOLD,
    assess_evidence,
    normalize_statement,
)
from medrag_eval.datasets import EvalItem, load_full_set, load_golden
from medrag_eval.metrics import Dataset, normalize_ground_truth, normalize_prediction

ROOT = Path(__file__).resolve().parents[2]
TARGET_ACCURACIES = (0.80, 0.85, 0.90, 0.95)
# Hand-labelled sample: (score, low, high, how many pairs). Entailment and
# contradiction bins take each statement's best passage for that score; "any" takes
# random pairs, most of them unrelated.
LABEL_BINS = (
    ("entailment", 0.9, 1.01, 50),
    ("entailment", 0.7, 0.9, 40),
    ("entailment", 0.5, 0.7, 25),
    ("entailment", 0.3, 0.5, 25),
    ("contradiction", 0.9, 1.01, 40),
    ("any", 0.0, 1.01, 40),
)


def final_iteration(row: dict[str, Any]) -> int:
    """0-based index of the iteration whose answer the run returned."""
    history = row["history"]
    for i, h in enumerate(history):
        if h["answer"] == row["final_answer"] and h["support_score"] == row["support_score"]:
            return i
    for i, h in enumerate(history):
        if h["answer"] == row["final_answer"]:
            return i
    return len(history) - 1


def score_row(
    row: dict[str, Any],
    item: EvalItem,
    retrieve: Callable[..., list[Any]],
    probabilities: Callable[[list[tuple[str, str]]], Sequence[Sequence[float]]],
    threshold: float = EVIDENCE_THRESHOLD,
) -> dict[str, Any]:
    dataset = Dataset(row["dataset"])
    index = final_iteration(row)
    exclude = [row["excluded_pmid"]] if row.get("excluded_pmid") else []
    passages = retrieve(row["history"][index]["query"], exclude_pmids=exclude)
    texts = [p.text for p in passages]
    statements = [s for s in row["rationale"] if s.strip()]
    claims = [normalize_statement(s) for s in statements]
    matrix = probabilities([(p, c) for c in claims for p in texts]) if claims and texts else []
    n = len(texts)
    scored = []
    for i, (statement, claim) in enumerate(zip(statements, claims, strict=True)):
        block = matrix[i * n : (i + 1) * n]
        entail = [float(r[ENTAILMENT]) for r in block]
        contra = [float(r[CONTRADICTION]) for r in block]
        scored.append(
            {
                "statement": statement,
                "claim": claim,
                "entailment": [round(e, 4) for e in entail],
                "contradiction": [round(c, 4) for c in contra],
            }
        )
    assessment = assess_evidence(statements, texts, lambda _pairs: matrix, threshold=threshold)
    return {
        "id": row["id"],
        "dataset": dataset.value,
        "answer": row["final_answer"],
        "gold": item.answer,
        "correct": normalize_prediction(row["final_answer"], dataset)
        == normalize_ground_truth(item.answer, dataset),
        "iteration": index + 1,
        "stored_support": row["support_score"],
        "status": assessment.status.value,
        "supported_fraction": assessment.supported_fraction,
        "passages": [{"pmid": p.pmid, "title": p.title, "text": p.text} for p in passages],
        "statements": scored,
    }


def answer_scores(result: dict[str, Any]) -> dict[str, float]:
    """Answer-level evidence scores; higher means better backed by the passages."""
    best = [max(s["entailment"], default=0.0) for s in result["statements"]]
    worst_contra = max(
        (max(s["contradiction"], default=0.0) for s in result["statements"]), default=0.0
    )
    return {
        "supported_fraction": result["supported_fraction"],
        "weakest_statement": min(best, default=0.0),
        "mean_statement": sum(best) / len(best) if best else 0.0,
        "strongest_statement": max(best, default=0.0),
        "weakest_minus_contradiction": min(best, default=0.0) - worst_contra,
    }


def risk_coverage(results: list[dict[str, Any]], score: str) -> dict[str, Any]:
    """Accuracy of the answers shown when only the top-scored ones are shown."""
    ranked = sorted(results, key=lambda r: -r["scores"][score])
    curve = []
    correct = 0
    for k, r in enumerate(ranked, start=1):
        correct += r["correct"]
        curve.append((k / len(ranked), correct / k, r["scores"][score]))
    table = {}
    for target in TARGET_ACCURACIES:
        # Largest coverage whose selective accuracy still reaches the target, cut
        # only between distinct scores (ties are shown or hidden together).
        best = None
        for i, (coverage, acc, s) in enumerate(curve):
            tie_follows = i + 1 < len(curve) and curve[i + 1][2] == s
            if acc >= target and not tie_follows:
                best = {"coverage": round(coverage, 4), "accuracy": round(acc, 4), "threshold": s}
        table[f"accuracy>={target:.2f}"] = best
    points = {}
    for c in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        k = max(1, round(c * len(ranked)))
        points[f"top {c:.0%}"] = round(curve[k - 1][1], 4)
    return {"max_coverage_at": table, "accuracy_of_top": points}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in Dataset:
        rows = [r for r in results if r["dataset"] == ds.value]
        if not rows:
            continue
        statuses: dict[str, dict[str, float]] = {}
        for status in sorted({r["status"] for r in rows}):
            group = [r for r in rows if r["status"] == status]
            statuses[status] = {
                "count": len(group),
                "share": round(len(group) / len(rows), 4),
                "accuracy": round(sum(r["correct"] for r in group) / len(group), 4),
            }
        out[ds.value] = {
            "answers": len(rows),
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            # Re-retrieval check: stored support should equal the recomputed one.
            "support_reproduced": round(
                sum(abs(r["stored_support"] - r["supported_fraction"]) < 1e-9 for r in rows)
                / len(rows),
                4,
            ),
            "by_status": statuses,
            "risk_coverage": {s: risk_coverage(rows, s) for s in rows[0]["scores"]},
        }
    return out


def label_sample(results: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Statement-passage pairs to judge by hand, sampled per bin (see LABEL_BINS)."""
    pairs = []
    for r in results:
        for s in r["statements"]:
            n = len(s["entailment"])
            if not n:
                continue
            best_e = max(range(n), key=lambda i: s["entailment"][i])
            best_c = max(range(n), key=lambda i: s["contradiction"][i])
            for j in range(n):
                pairs.append(
                    {
                        "id": r["id"],
                        "dataset": r["dataset"],
                        "claim": s["claim"],
                        "pmid": r["passages"][j]["pmid"],
                        "passage": r["passages"][j]["text"],
                        "entailment": s["entailment"][j],
                        "contradiction": s["contradiction"][j],
                        "best_entailment": j == best_e,
                        "best_contradiction": j == best_c,
                    }
                )
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    sample: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for field, low, high, count in LABEL_BINS:
        pool = [
            p
            for p in pairs
            if (field == "any" or (p[f"best_{field}"] and low <= p[field] < high))
            and (p["claim"], p["pmid"]) not in seen
        ]
        for p in rng.sample(pool, min(count, len(pool))):
            seen.add((p["claim"], p["pmid"]))
            sample.append({**p, "bin": f"{field} [{low}, {high})"})
    for n, p in enumerate(sample, start=1):
        p["pair"] = n
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", choices=("golden", "full"), default="golden")
    ap.add_argument("--nli-url", default=None)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data/indexes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out-name", default="grounding", help="output folder under the run, per verifier"
    )
    args = ap.parse_args()

    run_dir = ROOT / "eval/runs" / args.run
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    if any(r.get("live_pubmed") for r in rows):
        sys.exit("runs with live PubMed cannot be re-scored reproducibly")
    rows = [r for r in rows if r["stop_reason"] != "error"][: args.limit]
    items = (
        load_golden()
        if args.set == "golden"
        else [i for ds in Dataset for i in load_full_set(ds, args.data_dir)]
    )
    by_id = {i.id: i for i in items}

    from medrag_agent.runtime import build_components
    from medrag_settings import get_settings

    components = build_components(
        get_settings(), index_dir=args.index_dir, nli_url=args.nli_url, support_label="entailment"
    )
    retrieve = components.retriever.passages
    probabilities = components.nli.probabilities
    threshold = components.nli.support_threshold

    out_dir = run_dir / args.out_name
    out_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(
                lambda r: score_row(r, by_id[r["id"]], retrieve, probabilities, threshold), rows
            )
        )
    for r in results:
        r["scores"] = answer_scores(r)
    (out_dir / "statements.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), "utf-8"
    )
    summary = {
        "run": args.run,
        "verifier": getattr(components.nli, "version", "in-process"),
        "threshold": threshold,
        "datasets": summarise(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sample = label_sample(results, args.seed)
    (out_dir / "label_sample.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in sample), "utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
