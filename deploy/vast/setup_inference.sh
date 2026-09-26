#!/usr/bin/env bash
# Run ON a rented GPU host (Vast.ai) to serve NLI for long evaluation runs.
# Expects packages/settings, packages/core and services/inference copied to ~/medrag
# (see deploy/vast/README.md). No API keys or .env are needed or copied.
set -euo pipefail
cd ~/medrag
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv -q --python 3.12 .venv
uv pip install -q -p .venv/bin/python \
  -e packages/settings -e packages/core -e services/inference pytest
# GPU build of ONNX Runtime, with CUDA and cuDNN from pip.
uv pip uninstall -q -p .venv/bin/python onnxruntime
uv pip install -q -p .venv/bin/python "onnxruntime-gpu[cuda,cudnn]>=1.22"

.venv/bin/python - <<'PY'
import onnxruntime as ort
ort.preload_dlls()
print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "no CUDA provider"
PY

# Check the GPU against the research-work NLI fixture before serving anything.
MEDRAG_TEST_DEVICE=cuda .venv/bin/python -m pytest -q -m model -p no:cacheprovider \
  services/inference/tests/test_nli_model.py

# Serve on localhost only; the Jetson reaches it through an SSH tunnel.
INFERENCE_DEVICE=cuda nohup .venv/bin/python -m uvicorn medrag_inference.app:app \
  --host 127.0.0.1 --port 8001 > inference.log 2>&1 &
for _ in $(seq 60); do curl -sf localhost:8001/healthz && echo && exit 0; sleep 2; done
echo "service did not start; see ~/medrag/inference.log" >&2
exit 1
