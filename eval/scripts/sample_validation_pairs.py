"""Held-out statement-passage pairs for checking a verifier's chosen thresholds.

The thresholds in eval/grounding/verifier_benchmark.json were picked on the same
pairs they were measured on. This draws a fresh sample from the same run, stratified
by the chosen verifier's score (benchmark_verifiers.py --rescore output) and leaving
out every pair already labelled, to be judged by hand without seeing the scores.

    uv run python eval/scripts/sample_validation_pairs.py eval/runs/v2d-full/grounding \
        --verifier minicheck-roberta-large --out eval/grounding/validation_pairs.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# (low, high, how many): bins of the verifier's score on each statement's best passage.
BINS = ((0.5, 1.01, 40), (0.1878, 0.5, 40), (0.05, 0.1878, 30), (0.0, 0.05, 20))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grounding_dir", type=Path)
    ap.add_argument("--verifier", default="minicheck-roberta-large")
    ap.add_argument("--rescored", default="rescored_v2d.json")
    ap.add_argument("--labelled", type=Path, default=ROOT / "eval/grounding/verifier_labels.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (args.grounding_dir / "statements.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    scores = json.loads((args.grounding_dir / args.rescored).read_text("utf-8"))
    seen = {
        (r["claim"], r["pmid"])
        for r in map(json.loads, args.labelled.read_text("utf-8").splitlines())
    }
    pairs: list[dict[str, Any]] = []
    for r in rows:
        matrix = scores[r["id"]][args.verifier]
        for s, row in zip(r["statements"], matrix, strict=True):
            if not row:
                continue
            j = max(range(len(row)), key=lambda i: row[i])
            passage = r["passages"][j]
            if (s["claim"], passage["pmid"]) in seen:
                continue
            pairs.append(
                {
                    "id": r["id"],
                    "dataset": r["dataset"],
                    "claim": s["claim"],
                    "pmid": passage["pmid"],
                    "passage": passage["text"],
                    "score": row[j],
                }
            )
    rng = random.Random(args.seed)  # noqa: S311 - reproducible sampling, not security
    sample = []
    for low, high, count in BINS:
        pool = [p for p in pairs if low <= p["score"] < high]
        for p in rng.sample(pool, min(count, len(pool))):
            sample.append({**p, "bin": f"{args.verifier} [{low}, {high})"})
    rng.shuffle(sample)  # judged in random order, so bins cannot be guessed from position
    for n, p in enumerate(sample, start=1):
        p["pair"] = n
    args.out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in sample))
    print(f"{len(sample)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
