# deploy

Docker Compose files and the Caddy config for local runs; `vast/` has the GPU-host scripts.

## The whole system on one machine (`docker-compose.yml`)

```bash
make up              # build the images and start the stack
make ingest-sample   # load 2,000 PubMedQA records (SAMPLE_RECORDS), embed, index; restart the API
make eval-smoke      # end-to-end check through http://127.0.0.1:8090 (two LLM questions)
```

Open http://127.0.0.1:8090 for the web UI. `make stack-stats` shows memory per container;
`make down` stops the stack (data kept), `make clean` also deletes its database and index.

| Service | What it runs | Memory limit | Measured on the Jetson |
|---|---|---|---|
| `postgres` | PostgreSQL 17 + pgvector (internal only) | 768 MB | 62 MB |
| `migrate` | Alembic migrations, then exits | 512 MB | – |
| `inference` | embeddings + claim verifier on CPU | 2 GB | 1.2–1.4 GB |
| `api` | FastAPI agent service | 1.5 GB | 460 MB |
| `web` | Caddy: the static UI and a proxy for `/v1`, `/healthz`, `/readyz` | 128 MB | 41 MB |
| `worker` | corpus ingestion job (`make ingest-sample`) | 2 GB | while ingesting |

About 1.9 GB in all when answering questions, so it runs on an 8 GB Jetson Orin Nano with the
desktop apps closed. Images are built for the machine's own platform (arm64 on the Jetson,
amd64 on a laptop). Notes:

- **Secrets and data stay out of images** (`.dockerignore`); the API reads the LLM keys from
  `../.env` at run time. Only `127.0.0.1:8090` is published.
- **One LLM budget.** The API writes to the same spend ledger as local runs
  (`../data/llm_spend.json`), so the budget cap covers both.
- **Separate data.** The stack has its own database and index volumes; the development
  database and `data/indexes` are not touched.
- **Models.** On first start the services download BGE and the verifier into the Hugging Face
  cache. `make` reuses the host's `~/.cache/huggingface` when there is one (`HF_CACHE_DIR`).
- **Verifier.** Locally the inference service runs DeBERTa NLI (`VERIFIER=deberta-nli`), which
  fits in memory but is not the evaluated grounding setup; `/readyz` says so. The product's
  verifier, MiniCheck, needs a GPU host (next section) or a machine with more memory
  (`VERIFIER=minicheck` with `VERIFIER_ONNX_PATH` pointing to its exported ONNX file).
- **Not included yet:** Valkey and Phoenix from the plan. Nothing uses them so far (rate limits
  are in memory, tracing is not instrumented); they will be added with the features that need them.

### The stack with MiniCheck on a GPU host

1. Serve MiniCheck on a GPU host with `vast/setup_minicheck.sh` (see "Serving the MiniCheck
   verifier" in [vast/README.md](vast/README.md)).
2. Open the tunnel on the Docker bridge address, so containers can reach it (the loop
   reconnects dropped connections):

   ```bash
   while true; do
     ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
         -p <PORT> -L 172.17.0.1:18001:localhost:8001 root@<HOST>
     echo "tunnel dropped, reconnecting in 5 s"; sleep 5
   done
   ```

3. Point the API at it and check:

   ```bash
   make stack-minicheck   # recreates the api service with MEDRAG_NLI_URL=http://host.docker.internal:18001
   make eval-smoke        # /readyz should now say "minicheck + quotes"
   ```

   `make up` switches the API back to the local verifier.

## Development database only (`docker-compose.dev.yml`)

PostgreSQL + pgvector on `127.0.0.1:5432` for running the API, evaluations and tests from the
host (`docker compose -f deploy/docker-compose.dev.yml up -d`).
