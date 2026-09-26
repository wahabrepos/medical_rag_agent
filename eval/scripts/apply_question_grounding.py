"""Re-apply the current evidence gate to a finished run (no LLM calls).

Question grounding (medrag_core.evidence.ground_in_question) and the rule that
study aims are not evidence (states_aim) were added after some Step 7b runs. Both
only change the final assessment, not the agent loop, so applying them to the
recorded statements gives what the product would have shown: a statement is backed
by a study if it was supported (quoted runs: with a verified quote that is not an
aim sentence); the others are checked against the question with the same verifier;
then `is_grounded` decides. Writes predictions.question-grounded.jsonl
next to the run and prints the gated metrics.

    uv run --env-file .env python eval/scripts/apply_question_grounding.py \
        --run v3g-quotes-textbooks --nli-url http://localhost:18001
"""

import argparse
import json
from pathlib import Path

from medrag_core.evidence import (
    EvidenceAssessment,
    EvidenceStatus,
    StatementEvidence,
    ground_in_question,
    is_grounded,
    states_aim,
)
from medrag_eval.datasets import load_golden
from medrag_eval.metrics import Dataset, gated_metrics
from medrag_inference.client import RemoteNli

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--nli-url", required=True)
    args = ap.parse_args()

    run_dir = ROOT / "eval/runs" / args.run
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    items = {i.id: i for i in load_golden()}
    verifier = RemoteNli(args.nli_url)
    if any(r["nli"] != verifier.version for r in rows):
        raise SystemExit("the run used another verifier than the service serves")
    for r in rows:

        def backed(s: dict[str, object], quoted: bool = r["evidence_quotes"]) -> bool:
            if quoted:
                return isinstance(s["quote"], str) and not states_aim(s["quote"])
            return float(s["support"]) >= verifier.support_threshold  # type: ignore[arg-type]

        # Only whether a statement is supported matters here; the index is a placeholder.
        statements = [
            StatementEvidence(s["text"], s["support"], 0 if backed(s) else None, 0.0, None)
            for s in r["statements"]
        ]
        base = EvidenceAssessment(EvidenceStatus.NOT_SUPPORTED, 0.0, statements)
        grounded = ground_in_question(
            base,
            items[r["id"]].question,
            lambda pairs: verifier.probabilities(pairs).tolist(),
            threshold=verifier.support_threshold,
        )
        r["gated_shown"] = is_grounded(grounded)
        r["from_question"] = [s.from_question for s in grounded.statements]
    out = run_dir / "predictions.question-grounded.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    for ds in Dataset:
        subset = [r for r in rows if r["dataset"] == ds.value]
        if subset:
            truth = [items[r["id"]].answer for r in subset]
            print(ds.value, gated_metrics(subset, truth, ds))


if __name__ == "__main__":
    main()
