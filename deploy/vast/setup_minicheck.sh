#!/usr/bin/env bash
# Run ON a rented GPU host (Vast.ai) to serve the MiniCheck claim verifier (the product's
# verifier, VERIFIER=minicheck) for the API, the web UI or evaluation runs.
#
# Expects in ~/medrag (see deploy/vast/README.md; no API keys, .env or data are copied):
#   packages/settings packages/core services/inference deploy/vast
#   eval/grounding/validation_pairs.jsonl   (pairs for the export's PyTorch check)
#
# 1. Exports lytang/MiniCheck-RoBERTa-Large to ONNX in a throwaway PyTorch environment
#    (skipped when data/models/minicheck-roberta-large.onnx already exists) and refuses
#    to go on if the ONNX scores disagree with PyTorch.
# 2. Runs setup_inference.sh with VERIFIER=minicheck: GPU ONNX Runtime, model tests on the
#    GPU, then serves /healthz and /nli on the host's localhost:8001.
# Takes about 10 minutes on a fresh RTX 3060 host; needs about 15 GB of disk while exporting.
set -euo pipefail
cd ~/medrag
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

MODEL=data/models/minicheck-roberta-large.onnx
REPORT=data/models/export_report.json
mkdir -p data/models
if [ ! -f "$MODEL" ]; then
  [ -f eval/grounding/validation_pairs.jsonl ] || {
    echo "copy eval/grounding/validation_pairs.jsonl first (see deploy/vast/README.md)" >&2
    exit 1
  }
  uv venv -q --clear --python 3.12 ~/export-venv
  uv pip install -q -p ~/export-venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
  uv pip install -q -p ~/export-venv/bin/python "transformers>=4.44,<5" onnx \
    "onnxruntime-gpu[cuda,cudnn]" tokenizers huggingface-hub sentencepiece protobuf
  HF_HUB_DISABLE_XET=1 ~/export-venv/bin/python services/inference/scripts/export_minicheck.py \
    --out "$MODEL" --pairs eval/grounding/validation_pairs.jsonl --report "$REPORT"
  # Same bar as services/inference/tests/test_minicheck_model.py.
  python3 - "$REPORT" <<'PY' || { rm -f "$MODEL"; exit 1; }
import json, sys
report = json.load(open(sys.argv[1]))
for device, result in report["onnx"].items():
    ok = result["max_abs_diff"] < 0.002 and result["decision_flips"]["0.5"] == 0
    print(f"export check ({device}): max diff {result['max_abs_diff']}, "
          f"flips at 0.5: {result['decision_flips']['0.5']} -> {'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)
PY
  rm -rf ~/export-venv  # about 10 GB of PyTorch, not needed to serve the model
fi

VERIFIER=minicheck bash deploy/vast/setup_inference.sh
