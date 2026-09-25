"""Capture NLI outputs of the research-work code (read-only, Step 5).

Input: a JSON list of cases {id, rationale, passages} built from golden questions
(research-work rationale x passages retrieved by the parity profile). Output: the
full 3-class probabilities of cross-encoder/nli-deberta-v3-base computed exactly as
the research work did (PyTorch, CPU, batch 16, max_length 512), plus the result of
its verify_and_extract(), which reads column 2 as "entailment".

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \\
    <research-work>/venv/bin/python eval/capture/capture_step5.py \\
        --source <research-work> --repo . --inputs nli_inputs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    args = ap.parse_args()
    source, repo = args.source.resolve(), args.repo.resolve()
    cases = json.loads(args.inputs.read_text("utf-8"))
    os.chdir(source)
    sys.path.insert(0, str(source))

    import torch
    import torch.nn.functional as F
    import yaml
    from src.model import SelfReflectiveModule

    config = yaml.safe_load((source / "config_selfmedrag_mistral.yaml").read_text("utf-8"))
    module = SelfReflectiveModule(config)
    assert module.device.type == "cpu"

    out = []
    for case in cases:
        pairs = [(p, s) for s in case["rationale"] for p in case["passages"]]
        probs = []
        for start in range(0, len(pairs), module.nli_batch_size):
            batch = pairs[start : start + module.nli_batch_size]
            inputs = module.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                logits = module.model(**inputs).logits
            probs.extend(F.softmax(logits, dim=-1).tolist())
        score, unsupported = module.verify_and_extract(case["rationale"], case["passages"])
        out.append(
            {
                "id": case["id"],
                "rationale": case["rationale"],
                "passages": case["passages"],
                "probabilities": probs,
                "research_work_support_score": score,
                "research_work_unsupported": unsupported,
            }
        )

    target = repo / "services/inference/tests/fixtures/research_work/nli_cases.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "model": config["self_reflection"]["nli_model"],
                "id2label": {int(k): v for k, v in module.model.config.id2label.items()},
                "research_work_column": 2,
                "verification_threshold": module.verification_threshold,
                "cases": out,
            },
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    print(f"{len(out)} cases, {sum(len(c['probabilities']) for c in out)} pairs -> {target}")


if __name__ == "__main__":
    main()
