"""Compare claim verifiers on hand-labelled statement-passage pairs.

Reads eval/grounding/verifier_labels.jsonl (from analyze_grounding.py, judged by
hand: supported / not_enough / contradicted) and scores every pair with each
candidate model. A verifier is useful for the product only if its "supported" is
almost always right, so the report centres on precision: the largest recall
reached at precision >= 0.90 and >= 0.95, and the false-support rate on pairs
whose passage is unrelated to the claim.

Needs a GPU host with torch and transformers (see deploy/vast/README.md); it does
not use the workspace packages:

    python benchmark_verifiers.py verifier_labels.jsonl --out verifier_benchmark.json

With --rescore, it instead scores every statement-passage pair of an
analyze_grounding.py statements.jsonl with the --only candidates and writes the
scores per answer, for answer-level risk-coverage with that verifier.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
Scorer = Callable[[list[tuple[str, str]]], list[float]]

# name -> (kind, model id). "nli": 3-class NLI, score = P(entailment);
# "binary": 2-class supported/unsupported classifier, score = P(label 1).
CANDIDATES: dict[str, tuple[str, str]] = {
    "nli-deberta-v3-base (current)": ("nli", "cross-encoder/nli-deberta-v3-base"),
    "deberta-v3-base-mnli-fever-anli": ("nli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
    "deberta-v3-large-mnli-fever-anli-ling-wanli": (
        "nli",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    ),
    "pubmedbert-mnli-mednli": ("nli", "pritamdeka/PubMedBERT-MNLI-MedNLI"),
    "minicheck-roberta-large": ("binary", "lytang/MiniCheck-RoBERTa-Large"),
    "minicheck-deberta-v3-large": ("binary", "lytang/MiniCheck-DeBERTa-v3-Large"),
    "minicheck-flan-t5-large": ("t5", "lytang/MiniCheck-Flan-T5-Large"),
    "vectara-hhem-2.1-open": ("hhem", "vectara/hallucination_evaluation_model"),
}


def batched(pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    return [pairs[i : i + BATCH] for i in range(0, len(pairs), BATCH)]


def classifier_scorer(model_id: str, kind: str) -> Scorer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()
    if kind == "nli":
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        column = next(i for name, i in labels.items() if name.startswith("entail"))
    else:
        column = 1

    @torch.inference_mode()
    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for batch in batched(pairs):
            enc = tokenizer(
                [p for p, _ in batch],
                [c for _, c in batch],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            probs = model(**enc).logits.float().softmax(-1)
            out += probs[:, column].tolist()
        return out

    return score


def t5_scorer(model_id: str) -> Scorer:
    """MiniCheck-Flan-T5: P("1") vs P("0") as the first generated token."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(DEVICE).eval()
    # First decoding step: "1" is one token ("▁1"), "0" starts with "▁" then "0".
    zero = tokenizer("0", add_special_tokens=False).input_ids[0]
    one = tokenizer("1", add_special_tokens=False).input_ids[0]

    @torch.inference_mode()
    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for batch in batched(pairs):
            enc = tokenizer(
                [f"premise: {p} hypothesis: {c}" for p, c in batch],
                truncation=True,
                max_length=1024,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            start = torch.full((len(batch), 1), model.config.decoder_start_token_id, device=DEVICE)
            logits = model(**enc, decoder_input_ids=start).logits[:, 0, [zero, one]]
            out += logits.float().softmax(-1)[:, 1].tolist()
        return out

    return score


def hhem_scorer(model_id: str) -> Scorer:
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
        .to(DEVICE)
        .eval()
    )

    @torch.inference_mode()
    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for batch in batched(pairs):
            out += [float(x) for x in model.predict(batch)]
        return out

    return score


def build(kind: str, model_id: str) -> Scorer:
    if kind == "t5":
        return t5_scorer(model_id)
    if kind == "hhem":
        return hhem_scorer(model_id)
    return classifier_scorer(model_id, kind)


def metrics(scores: list[float], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [r["label"] == "supported" for r in rows]
    positives = sum(gold)
    ranked = sorted(zip(scores, gold, strict=True), key=lambda x: -x[0])
    # Average precision and the best recall at each precision floor.
    tp = 0
    ap = 0.0
    best: dict[float, dict[str, float] | None] = {0.90: None, 0.95: None}
    for k, (s, g) in enumerate(ranked, start=1):
        tp += g
        if g:
            ap += tp / k
        tie_follows = k < len(ranked) and ranked[k][0] == s
        if tie_follows:
            continue
        for floor in best:
            if tp / k >= floor:
                best[floor] = {
                    "threshold": round(s, 4),
                    "recall": round(tp / positives, 4),
                    "precision": round(tp / k, 4),
                }
    at_half = [s >= 0.5 for s in scores]
    tp5 = sum(p and g for p, g in zip(at_half, gold, strict=True))
    unrelated = [
        s
        for s, r in zip(scores, rows, strict=True)
        if r["bin"].startswith(("any", "contradiction")) and r["label"] != "supported"
    ]
    return {
        "average_precision": round(ap / positives, 4),
        "at_0.5": {
            "precision": round(tp5 / max(1, sum(at_half)), 4),
            "recall": round(tp5 / positives, 4),
        },
        "max_recall_at_precision_0.90": best[0.90],
        "max_recall_at_precision_0.95": best[0.95],
        "false_support_on_unrelated_at_0.5": round(
            sum(s >= 0.5 for s in unrelated) / max(1, len(unrelated)), 4
        ),
    }


def rescore(path: Path, names: list[str], out: Path) -> None:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    pairs = [(p["text"], s["claim"]) for r in rows for s in r["statements"] for p in r["passages"]]
    result: dict[str, dict[str, list[list[float]]]] = {r["id"]: {} for r in rows}
    for name in names:
        kind, model_id = CANDIDATES[name]
        started = time.perf_counter()
        scores = iter(build(kind, model_id)(pairs))
        for r in rows:
            result[r["id"]][name] = [
                [round(next(scores), 4) for _ in r["passages"]] for _ in r["statements"]
            ]
        print(name, f"{len(pairs)} pairs in {time.perf_counter() - started:.0f}s", flush=True)
        torch.cuda.empty_cache()
    out.write_text(json.dumps(result) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("verifier_benchmark.json"))
    ap.add_argument("--only", nargs="*", default=None, help="candidate names to run")
    ap.add_argument("--rescore", type=Path, default=None, help="statements.jsonl to re-score")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore, args.only or [], args.out)
        return

    rows = [json.loads(line) for line in args.labels.read_text("utf-8").splitlines() if line]
    pairs = [(r["passage"], r["claim"]) for r in rows]
    sanity = [
        ("Metformin lowered HbA1c in adults with type 2 diabetes.", "Metformin lowers HbA1c."),
        ("Metformin lowered HbA1c in adults with type 2 diabetes.", "Metformin raises HbA1c."),
        (
            "Metformin lowered HbA1c in adults with type 2 diabetes.",
            "Tetanus is caused by bacteria.",
        ),
    ]
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        "pairs": len(rows),
        "supported": sum(r["label"] == "supported" for r in rows),
        "models": {},
    }
    for name, (kind, model_id) in CANDIDATES.items():
        if args.only and name not in args.only:
            continue
        try:
            scorer = build(kind, model_id)
            started = time.perf_counter()
            scores = scorer(pairs)
            elapsed = time.perf_counter() - started
            result = metrics(scores, rows)
            result.update(
                {
                    "model": model_id,
                    "pairs_per_second": round(len(pairs) / elapsed, 1),
                    "sanity_entail_contra_unrelated": [round(s, 3) for s in scorer(sanity)],
                    "scores": [round(s, 4) for s in scores],
                }
            )
        except Exception as exc:  # one broken candidate must not stop the others
            result = {"model": model_id, "error": f"{type(exc).__name__}: {exc}"}
        report["models"][name] = result
        print(name, json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
