# medical_rag_agent

A cloud-hosted **agentic RAG** service for evidence-grounded medical question answering. It migrates the *Self-verifying Clinical Reasoning in a Loop MedRAG* (Self-MedRAG) research-work prototype off a local NVIDIA Jetson edge device and into the cloud.

The agent retrieves PubMed evidence with hybrid search, generates a JSON answer with a list of rationale statements, and checks each statement against the evidence with an NLI model. When support is too low, it refines the query and tries again.

> ⚠️ **Research use only. Not for clinical decisions.** Answers can be wrong even when they cite sources.

## Status

Early development. The workspace, tooling and the research-work reference data are in place. The core Self-MedRAG logic (prompts, JSON parsing, BM25 tokenizer, RRF fusion, NLI verification and the loop's stop rules) is ported to `packages/core`, and the data layer (PostgreSQL + pgvector schema, corpus ingestion, hybrid retrieval) reproduces the research-work retrieval: on 200 reference questions the fused top-5 overlap is 0.996 and BM25 scores are bit-identical. The first milestone is a **parity build** that reproduces the research-work results before any behaviour changes:

| System (research work) | MedQA | PubMedQA |
|---|---|---|
| Self-MedRAG + Mistral-small | 71.30% | 75.96% |

## Origin: the research-work prototype

Before this migration, Self-MedRAG ran as a single-process Python prototype on an edge device. The algorithm was the same (retrieve → generate → NLI-verify → refine), but it was written as a hand-coded loop rather than an agent graph, and it was used from the command line only.

### Previous stack

| Layer | Research-work prototype |
|---|---|
| Hardware | NVIDIA Jetson Orin Nano Super, 8 GB unified CPU/GPU memory (JetPack 6); rented Vast.ai RTX 3090 (24 GB) for the larger-model runs |
| Language / ML runtime | Python 3.10, PyTorch, Hugging Face Transformers, Accelerate, bitsandbytes (4-bit NF4 quantisation) |
| Generator (local) | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit on the Jetson, BF16 on the RTX 3090), `Qwen2.5-7B-Instruct` and `Qwen2.5-14B-Instruct` (4-bit NF4, RTX 3090) |
| Generator (API) | Mistral `mistral-small-latest` via the `mistralai` SDK, with custom retry and back-off |
| Verifier | `cross-encoder/nli-deberta-v3-base`, PyTorch (GPU when memory allowed, otherwise CPU) |
| Dense retrieval | `BAAI/bge-small-en-v1.5` (CLS pooling, 384 dimensions) + FAISS `IndexFlatIP`, cached to disk |
| Sparse retrieval | `rank_bm25` (BM25Okapi, k1 = 1.5, b = 0.75) with a custom medical stopword list |
| Fusion | Reciprocal Rank Fusion (k = 60), top 5 passages |
| Corpus | 10,000 PubMed abstracts (from the 203k-record PubMedQA abstract set), held in memory as plain strings |
| Benchmarks | MedQA (USMLE, 1,000 questions) and PubMedQA (890 questions) |
| Orchestration | Hand-written iterative loop (`Trainer.run()`): support threshold θ = 0.7, up to 3 iterations |
| Configuration | YAML files per experiment, `.env` for keys |
| Interface | CLI scripts and an interactive terminal chat; no API server or UI |
| Operations | Local log files and JSON checkpoints; no containers, IaC or CI/CD |

### Results by generator

| System | MedQA | PubMedQA | Ran on |
|---|---|---|---|
| BM25-only baseline (Mistral-small) | 72.00% | 63.26% | API |
| Dense BGE baseline (Mistral-small) | 72.30% | 79.10% | API |
| Hybrid RRF baseline (Mistral-small) | 71.40% | 76.40% | API |
| Self-MedRAG + Qwen2.5-1.5B | 37.40% | 51.35% | Jetson |
| Self-MedRAG + Qwen2.5-7B (NF4) | 46.20% | 70.22% | RTX 3090 |
| Self-MedRAG + Qwen2.5-14B (NF4) | 59.90% | 75.17% | RTX 3090 |
| **Self-MedRAG + Mistral-small** | **71.30%** | **75.96%** | **API** |

### Why it moved to the cloud

- **Memory limits:** local generators competed with the embedding and NLI models for 8 GB of shared memory, which caused out-of-memory errors, corrupted CUDA contexts, thermal shutdowns and reboots.
- **Accuracy:** the best result came from the API generator, which needs no local GPU at all.
- **Serving:** the prototype was single-user and CLI-only. Serving it to users needs an API, a UI, persistent storage and automated deployment.

## Cloud tech stack

What the migrated service uses.

| Layer | Choice |
|---|---|
| API | FastAPI (SSE streaming) |
| Agent | LangGraph, checkpoints in PostgreSQL |
| LLM | LiteLLM → Groq `openai/gpt-oss-120b` (Mistral-small is not on Mistral's free plan) |
| Embeddings | `BAAI/bge-small-en-v1.5`, self-hosted on CPU (ONNX) |
| Verifier | `cross-encoder/nli-deberta-v3-base`, CPU (ONNX fp32) |
| Retrieval | PostgreSQL + pgvector (HNSW) + BM25, fused with RRF |
| Frontend | Next.js |
| Infra | AWS (single VM + RDS PostgreSQL), OpenTofu, GitHub Actions, Docker |
| Observability | Sentry, Arize Phoenix |

No component needs a GPU.

## Architecture

### System overview

```mermaid
flowchart LR
    U[Clinician / researcher] --> FE[Next.js UI<br/>SSE stream, citations, support scores]
    FE -->|HTTPS| API[FastAPI<br/>auth · rate limit · /ask · /runs/:id]
    API --> G[LangGraph Self-MedRAG agent]
    G --> LLM[LiteLLM gateway]
    LLM --> M1[Mistral-small API<br/>primary: research-work parity]
    LLM --> M2[Groq<br/>fast fallback]
    LLM -. optional .-> M3[vLLM on rented GPU<br/>Qwen2.5-14B / 32B]
    G --> INF[CPU inference svc<br/>BGE-small embed · DeBERTa NLI · reranker]
    G --> PG[(PostgreSQL + pgvector<br/>chunks · checkpoints · runs)]
    G -. tool .-> PM[PubMed E-utilities<br/>live search]
    ING[Ingestion worker] --> INF
    ING --> PG
    API -. traces .-> OBS[Phoenix / Langfuse]
    API -. errors .-> SEN[Sentry]
```

### Agent workflow

The research-work loop, rebuilt as a LangGraph graph and extended so that the self-verification loop actually iterates when evidence is weak.

```mermaid
flowchart TD
    S([START]) --> C[classify_question<br/>MCQ · yes/no · open]
    C --> R[retrieve_hybrid<br/>pgvector + BM25 → RRF]
    R --> GR{grade_retrieval<br/>enough evidence?}
    GR -- no, iter < max --> RW[rewrite / decompose query]
    GR -- no local evidence --> PM[pubmed_search tool]
    PM --> R
    RW --> R
    GR -- yes --> GEN[generate<br/>JSON: answer, rationale, citations]
    GEN --> V[verify_nli<br/>rationale ⇐ passages<br/>answer ⇐ rationale+passages]
    V --> D{support ≥ θ ?}
    D -- yes --> F[finalize + safety check]
    D -- no, improving, iter < max --> RF[refine_query<br/>from unsupported statements]
    RF --> R
    D -- no, stalled or max iter --> F
    F --> E([END])
```

### Mapping from the research-work code

File paths refer to the original `self-medrag-production` project.

| Node | Reused from |
|---|---|
| `retrieve_hybrid` | `RetrievalModule.retrieve()` + `fuse_results()` (`src/retrieval.py`); storage moves to PostgreSQL |
| `generate` | `_construct_prompt()` + `_parse_json_output()` (`src/model.py`) + a LiteLLM call |
| `verify_nli` | `verify_and_extract()` (`src/model.py`), extended to also verify the answer |
| decision edge | Stop rules in `Trainer.run()` (`src/trainer.py`): θ, `min_improvement`, `max_iterations`, `max_time_seconds` |
| `refine_query` | `_refine_query()` (`src/trainer.py`): structured, decomposition or concatenation strategy |
| `finalize` | The "best of history" fallback from `Trainer.run()` |
| New nodes | `classify_question`, `grade_retrieval`, `pubmed_search`, safety check, `PostgresSaver` checkpointing |

### Agent state

Fields of the graph state (`TypedDict`): `question`, `question_type`, `options`, `current_query`, `iteration`, `passages[{chunk_id, pmid, title, text, score}]`, `answer`, `rationale[]`, `citations[chunk_id]`, `confidence`, `support_score`, `answer_supported`, `unsupported[]`, `history[]`, `llm_calls`, `started_at`.

## Repository layout

```
packages/settings/    typed runtime settings (pydantic-settings)
packages/core/        framework-free Self-MedRAG logic
packages/db/          PostgreSQL + pgvector schema (SQLAlchemy)
packages/search/      BM25 index, pgvector search, hybrid retriever
packages/agent/       LangGraph agent
apps/api/             FastAPI service
apps/web/             Next.js UI
services/inference/   embedding + NLI service (CPU)
workers/ingest/       corpus ingestion jobs
eval/                 golden sets, reference results, evaluation gate
db/                   Alembic migrations
infra/tofu/           OpenTofu modules and environments
deploy/               Docker Compose and Caddy config
```

## Development

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you) and Docker.

```bash
cp .env.example .env      # then fill in MISTRAL_API_KEY, GROQ_API_KEY, HF_TOKEN
make install              # uv sync --all-packages
make hooks                # install pre-commit hooks (ruff, mypy, gitleaks, ...)
make check                # lint + type-check + tests
```

Integration tests need PostgreSQL with pgvector:

```bash
docker compose -f deploy/docker-compose.dev.yml up -d
TEST_DATABASE_URL=postgresql+psycopg://medrag:medrag@localhost:5432/medrag \
  uv run pytest -m integration        # uses throwaway databases, never the dev one
```

### Corpus and retrieval

```bash
uv run --env-file .env alembic -c db/alembic.ini upgrade head
uv run --env-file .env medrag-ingest all --limit 10000 --reset       # research-work corpus size
uv run --env-file .env --with datasets python eval/scripts/fetch_eval_sets.py
uv run --env-file .env python eval/scripts/check_retrieval_parity.py  # needs >= 0.8, got 0.996
```

The corpus is PubMedQA (`pqa_labeled` + `pqa_unlabeled`): 203,429 abstract sections with their
PubMed IDs, exactly the research-work corpus. Two chunking profiles are stored side by side:
`parity` (the research-work chunker, used for the parity build) and `standard` (token-aware
chunks of about 350 tokens over whole abstracts).

> **Known limitation.** `pqa_labeled` contains the abstracts of the PubMedQA evaluation
> questions, so for about 98% of those questions the source abstract is retrieved in the top 5.
> PubMedQA accuracy is therefore "open-book", in the research work and here. A leakage-free
> evaluation will exclude those abstracts.



### Inference service

Embeddings and NLI run on CPU in one small service (no GPU needed):

```bash
uv run --env-file .env uvicorn medrag_inference.app:app --port 8001
# GET /healthz · POST /embed {"texts": [...]} · POST /nli {"pairs": [[premise, hypothesis], ...]}
uv run --env-file .env pytest -m model services/inference   # model checks (downloads weights)
```

- **NLI labels.** The model's outputs are `contradiction`, `entailment`, `neutral` (in that
  order). The research work read the third column as entailment, so its "support score" was
  really the probability of *neutral*. `/nli` returns all three probabilities; the parity build
  keeps the research-work reading, and the corrected verifier (entailment) is evaluated
  separately. On 16 golden questions the mean support is 1.00 with the research-work column
  and 0.06 with entailment.
- **fp32, not int8.** fp32 ONNX reproduces the research-work PyTorch scores exactly. No int8
  variant (published exports or our own dynamic quantisation, full or partial) kept decisions
  stable; see `services/inference/reports/nli_variants_x86_zen4.json`. fp32 needs about 4 s for
  25 pairs on 2 Zen 4 threads; repeated pairs are served from an in-memory cache.

### Agent and evaluation

The Self-MedRAG loop runs as a LangGraph graph (`packages/agent`): retrieve → generate →
verify → decide, refining the query from unsupported statements. It is tested to behave
exactly like the research-work loop.

```bash
uv run --env-file .env python eval/scripts/run_agent_eval.py --set golden --run v1-golden
uv run --env-file .env python eval/scripts/run_agent_eval.py --set golden --run v1-golden --report-only
```

Runs are resumable (answers are appended to `eval/runs/<run>/predictions.jsonl`) and stop
cleanly when the provider's daily quota is used up (exit code 3).

**Generator.** The research work used Mistral `mistral-small-latest`, which Mistral's free plan
no longer serves, so the first build uses Groq `openai/gpt-oss-120b` (free tier: 1,000
requests/day, 8,000 tokens/minute). It is a reasoning model: it gets `reasoning_effort="low"`
and 2,048 tokens of headroom, because with the research work's 400 tokens the hidden reasoning
used up the budget and answers came back empty. Because the generator differs, accuracy is not
directly comparable with the research work's 71.30% / 75.96%.

**v1 baseline** (research-work pipeline with gpt-oss-120b, golden 150):

| | MedQA (75) | PubMedQA (75) |
|---|---|---|
| Agent | 62.7% | 77.3% |
| Research work, same questions | 69.3% | 76.0% |
| Questions needing more than one iteration | 2.7% | 1.3% |

16 of the 75 MedQA answers were "insufficient evidence" (the system prompt asks for this when
the context does not support a claim); on the other 59 the agent was right 47 times.

### v2 experiments (golden 150, gpt-oss-120b)

Each change is measured against the v1 baseline on the same 150 questions:

| Run | Change | MedQA | PubMedQA | MedQA "insufficient evidence" | Loop iterates |
|---|---|---|---|---|---|
| v1 | research-work pipeline | 62.7% | 77.3% | 16 | 1–3% |
| v2a | corrected verifier (NLI *entailment*) | 32.0% | 78.7% | 49 | ~100% |
| v2b | v2a + keep the best-supported answer, never replace an answer with a refusal | 64.0% | 77.3% | 10 | ~100% |
| v2c | v2b + multiple-choice questions must name an option | **80.0%** | 77.3% | 0 | ~100% |
| v2d | v2c + read answer fields from invalid JSON (0 raw-JSON answers, was 6) | 77.3% | 77.3% | 0 | ~100% |
| research work | Mistral-small, same questions | 69.3% | 76.0% | – | 0% |

- With the corrected verifier the loop actually iterates, but the research-work rules then
  return the latest answer, which often backs off to "insufficient evidence" (v2a). Keeping
  the best-supported answer and preferring real answers fixes that (v2b).
- The multiple-choice constraint removes the refusals (v2c). Support stays low on MedQA
  (about 0.03): the PubMedQA-based corpus rarely backs USMLE-style reasoning, so many MedQA
  answers rely on the model's own knowledge. Low-support answers must be flagged to users.
- Differences of a few points on 150 questions are within noise (v2c and v2d differ by two
  MedQA questions); a full-set run confirms them. v2d is the configuration for the full run.

**Full evaluation of v2d** (all 1,000 MedQA and 890 PubMedQA questions; NLI on a rented GPU):

| | Agent (v2d, gpt-oss-120b) | Research work (Self-MedRAG + Mistral-small) |
|---|---|---|
| MedQA | **84.9%** | 71.3% |
| PubMedQA | 74.8% | 76.0% |
| Questions needing more than one iteration | 99.3% / 93.9% | about 1% |

MedQA improves by 13.6 points (about ±2 points of uncertainty at this size); PubMedQA is on
par (−1.1 points, within noise). MedQA support stays low (0.03), so those answers rely largely
on the model's own knowledge. PubMedQA is still open-book: the gold abstracts are retrievable.

## License

MIT. See [LICENSE](LICENSE).
