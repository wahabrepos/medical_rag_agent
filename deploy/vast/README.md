# deploy/vast

Run a long evaluation with NLI on a rented GPU while everything else (retrieval,
PostgreSQL, LLM calls and API keys) stays on the local machine. NLI on a small CPU
is the slow part of a run; on a GPU it is almost free.

1. Rent a GPU host (e.g. RTX 3060/3090, reliability >= 99%). Note its SSH command.
2. Copy the inference code (no data, no secrets):

   ```bash
   rsync -a --partial -e "ssh -p <PORT>" --relative \
     packages/settings packages/core services/inference deploy/vast \
     root@<HOST>:medrag/
   ssh -p <PORT> root@<HOST> 'bash ~/medrag/deploy/vast/setup_inference.sh'
   ```

   The script installs the GPU build of ONNX Runtime, checks the GPU against the
   research-work NLI fixture, and serves `/nli` on the host's localhost:8001.
3. Open a tunnel and run the evaluation locally:

   ```bash
   ssh -N -p <PORT> -L 8001:localhost:8001 root@<HOST> &
   uv run --env-file .env python eval/scripts/run_agent_eval.py --set full --run <name> \
     --nli-url http://localhost:8001 <experiment options>
   ```

   If local port 8001 is taken, use another one (e.g. `-L 18001:localhost:8001` and
   `--nli-url http://localhost:18001`). If the tunnel drops, the run stops cleanly;
   reopen the tunnel and run the same command again to resume.
4. Destroy the instance when the run is finished.
