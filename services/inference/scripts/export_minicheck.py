"""Export the MiniCheck-RoBERTa-Large claim verifier to ONNX and check it.

The Hugging Face repo only has PyTorch weights, while the inference service runs
ONNX Runtime without torch. This exports the pinned revision (fp32, opset 17,
dynamic batch and sequence length) and compares the ONNX probabilities, computed
with the same `tokenizers` pipeline the service uses, with PyTorch + transformers
on hand-labelled pairs: largest difference and decisions flipped at the given
thresholds.

Run it once where torch is available (e.g. the GPU host's benchmark venv):

    python export_minicheck.py --out minicheck-roberta-large.onnx \
        --pairs verifier_labels.jsonl --report export_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "lytang/MiniCheck-RoBERTa-Large"
REVISION = "74c8919647e61ed0f71bc177d94f10930f090068"
MAX_LENGTH = 512
THRESHOLDS = (0.1878, 0.5)


def export(out: Path) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    enc = tokenizer(["a passage"], ["a claim"], return_tensors="pt")
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def onnx_scores(path: Path, pairs: list[tuple[str, str]], device: str) -> np.ndarray:
    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL_ID, "tokenizer.json", revision=REVISION))
    tokenizer.enable_truncation(max_length=MAX_LENGTH, strategy="only_first")
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    out = []
    for passage, claim in pairs:
        e = tokenizer.encode(passage, claim)
        logits = session.run(
            None,
            {
                "input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64),
            },
        )[0][0]
        p = np.exp(logits - logits.max())
        out.append(p[1] / p.sum())
    return np.array(out)


@torch.inference_mode()
def torch_scores(pairs: list[tuple[str, str]], device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION)
    model = model.to(device).eval()
    out = []
    for passage, claim in pairs:
        enc = tokenizer(
            passage, claim, truncation="only_first", max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        out.append(model(**enc).logits.float().softmax(-1)[0, 1].item())
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="jsonl with passage and claim")
    ap.add_argument("--report", type=Path, default=Path("export_report.json"))
    args = ap.parse_args()

    if not args.out.exists():
        export(args.out)
    rows = [json.loads(line) for line in args.pairs.read_text("utf-8").splitlines() if line]
    pairs = [(r["passage"], r["claim"]) for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = torch_scores(pairs, device)
    report = {"model": MODEL_ID, "revision": REVISION, "pairs": len(pairs), "onnx": {}}
    for provider in ("cpu", "cuda") if device == "cuda" else ("cpu",):
        scores = onnx_scores(args.out, pairs, provider)
        diff = np.abs(scores - reference)
        report["onnx"][provider] = {
            "max_abs_diff": round(float(diff.max()), 6),
            "decision_flips": {
                str(t): int(np.sum((scores >= t) != (reference >= t))) for t in THRESHOLDS
            },
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
