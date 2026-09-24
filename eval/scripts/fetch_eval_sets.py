"""Download the full research-work evaluation sets and verify them against the manifest.

Rebuilds the exact questions the research work evaluated (1,000 MedQA + 890 PubMedQA)
from Hugging Face, formats them the same way, and checks every question hash and
answer against eval/reference/eval_sets_manifest.json. The output goes to data/eval/
(gitignored):

    uv run --env-file .env --with datasets python eval/scripts/fetch_eval_sets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from medrag_eval.datasets import format_medqa_question

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/reference/eval_sets_manifest.json"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_medqa() -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for item in load_dataset("openlifescienceai/medqa", split="test"):
        inner = item.get("data", item)
        raw = inner.get("Options", inner.get("options", {}))
        if isinstance(raw, dict):
            options = [str(raw[k]).strip() for k in sorted(raw)]
        else:
            options = [str(o).strip() for o in raw]
        options = [o for o in options if o]
        question = str(inner.get("Question", inner.get("question", ""))).strip()
        rows.append(
            {
                "question": format_medqa_question(question, options),
                "options": options,
                "answer": str(inner.get("Correct Option", "")).strip().upper(),
                "answer_text": str(inner.get("Correct Answer", "")).strip(),
                "dataset_type": "medqa",
            }
        )
    return rows


def load_pubmedqa() -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [
        {
            "question": str(item["question"]).strip(),
            "answer": str(item["final_decision"]).strip(),
            "pubid": item.get("pubid"),
            "dataset_type": "pubmedqa",
        }
        for item in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, loader in (("medqa", load_medqa), ("pubmedqa", load_pubmedqa)):
        source = loader()
        out_rows = []
        for ref in manifest[name]:
            row = dict(source[ref["source_index"]])
            same_question = sha256(row["question"]) == ref["question_sha256"]
            if not (same_question and row["answer"] == ref["answer"]):
                failures += 1
                print(f"MISMATCH {ref['id']}", file=sys.stderr)
            out_rows.append({"id": ref["id"], **row})
        path = args.out / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
        print(f"{name}: {len(out_rows)} questions -> {path.relative_to(ROOT)}")
    if failures:
        print(f"{failures} items did not match the manifest", file=sys.stderr)
        return 1
    print("all items match the research-work manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
