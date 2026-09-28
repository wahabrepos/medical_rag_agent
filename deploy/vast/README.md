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

## Serving the MiniCheck verifier (API, web UI, live demos)

The product's claim verifier (`VERIFIER=minicheck`, see "Step 7b" in the root README) is
too large for the Jetson, so it runs on the GPU host while the API and the web UI run locally.

1. Rent a GPU host (an RTX 3060 is enough) and copy the code plus the pairs the export is
   checked on:

   ```bash
   rsync -a --partial -e "ssh -p <PORT>" --relative \
     packages/settings packages/core services/inference deploy/vast \
     eval/grounding/validation_pairs.jsonl root@<HOST>:medrag/
   ssh -p <PORT> root@<HOST> 'bash ~/medrag/deploy/vast/setup_minicheck.sh'
   ```

   `setup_minicheck.sh` exports MiniCheck to ONNX (checked against PyTorch; the script stops
   if they disagree), runs the model tests on the GPU and serves the verifier on the host's
   localhost:8001. About 10 minutes on a fresh host; running it again only restarts the service.
2. Open a tunnel that notices dropped connections and reconnects (a plain `ssh -N` can hang
   silently, and the API then waits on it):

   ```bash
   while true; do
     ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
         -p <PORT> -L 18001:localhost:8001 root@<HOST>
     echo "tunnel dropped, reconnecting in 5 s"; sleep 5
   done
   ```

   `curl localhost:18001/healthz` should report `MiniCheck-RoBERTa-Large` and threshold 0.5.
3. Start the API with the product settings (the database container must be running):

   ```bash
   APP_ENV=local NLI_URL=http://localhost:18001 EVIDENCE_QUOTES=true \
   ANSWER_POLICY=evidence_gated CORS_ORIGINS=http://localhost:3000 \
   uv run --env-file .env uvicorn medrag_api.app:app --port 8000
   ```

   `curl localhost:8000/readyz` should show `"grounding": "minicheck + quotes"`.
4. Start the web UI and open http://localhost:3000:

   ```bash
   cd apps/web && NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm dev
   ```

   Each question costs about €0.0005 of LLM spend (the API stops before the budget is used up).
5. Stop the three processes and destroy the instance when you are done.
