"""Answer-level risk-coverage with re-scored verifiers.

Combines analyze_grounding.py's statements.jsonl (answers, correctness, the current
verifier's scores) with benchmark_verifiers.py --rescore output (other verifiers'
scores for the same statement-passage pairs) and reports, per dataset and verifier,
how many answers can be shown at a given accuracy when only the best-supported are.

    uv run python eval/scripts/verifier_risk_coverage.py eval/runs/v2d-full/grounding
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from analyze_grounding import risk_coverage

CURRENT = "nli-deberta-v3-base (current)"
# Pair-level support thresholds. Current and DeBERTa-large: precision >= 0.95 on
# eval/grounding/verifier_labels.jsonl; MiniCheck: its default 0.5, which held 100%
# precision on the held-out eval/grounding/validation_pairs.jsonl (0.188 only 90%).
SUPPORT_THRESHOLDS = {
    CURRENT: 0.9326,
    "minicheck-roberta-large": 0.5,
    "deberta-v3-large-mnli-fever-anli-ling-wanli": 0.9914,
}


def answer_scores(matrix: list[list[float]], threshold: float) -> dict[str, float]:
    best = [max(row, default=0.0) for row in matrix]
    if not best:
        return {"supported_fraction": 0.0, "weakest_statement": 0.0, "mean_statement": 0.0}
    return {
        "supported_fraction": sum(b >= threshold for b in best) / len(best),
        "weakest_statement": min(best),
        "mean_statement": sum(best) / len(best),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grounding_dir", type=Path)
    ap.add_argument("--rescored", default="rescored_v2d.json")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (args.grounding_dir / "statements.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    rescored = json.loads((args.grounding_dir / args.rescored).read_text("utf-8"))
    verifiers = [CURRENT, *next(iter(rescored.values())).keys()]
    report: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        report[dataset] = {"answers": len(subset)}
        for name in verifiers:
            scored = []
            for r in subset:
                matrix = (
                    [s["entailment"] for s in r["statements"]]
                    if name == CURRENT
                    else rescored[r["id"]][name]
                )
                scored.append(
                    {
                        "correct": r["correct"],
                        "scores": answer_scores(matrix, SUPPORT_THRESHOLDS[name]),
                    }
                )
            report[dataset][name] = {
                score: risk_coverage(scored, score) for score in scored[0]["scores"]
            }
    out = args.grounding_dir / "verifier_risk_coverage.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for dataset, by_verifier in report.items():
        print(f"== {dataset} ({by_verifier['answers']} answers)")
        for name, by_score in by_verifier.items():
            if name == "answers":
                continue
            for score, rc in by_score.items():
                cov = {
                    k.split(">=")[1]: (v["coverage"] if v else None)
                    for k, v in rc["max_coverage_at"].items()
                }
                print(f"  {name[:28]:28} {score:18} max coverage at accuracy {cov}")


if __name__ == "__main__":
    main()
