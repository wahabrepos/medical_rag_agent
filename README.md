# medical_rag_agent

A cloud-hosted **agentic RAG** service for evidence-grounded medical question answering. It migrates the *Self-verifying Clinical Reasoning in a Loop MedRAG* (Self-MedRAG) research-work prototype off a local NVIDIA Jetson edge device and into the cloud.

The agent retrieves PubMed evidence with hybrid search, generates a JSON answer with a list of rationale statements, and checks each statement against the evidence with an NLI model. When support is too low, it refines the query and tries again.

> ⚠️ **Research use only. Not for clinical decisions.** Answers can be wrong even when they cite sources.

## Status

Early development. The workspace, tooling and the research-work reference data are in place, and an API and a web UI serve the agent. The core Self-MedRAG logic (prompts, JSON parsing, BM25 tokenizer, RRF fusion, NLI verification and the loop's stop rules) is ported to `packages/core`, and the data layer (PostgreSQL + pgvector schema, corpus ingestion, hybrid retrieval) reproduces the research-work retrieval: on 200 reference questions the fused top-5 overlap is 0.996 and BM25 scores are bit-identical. The first milestone is a **parity build** that reproduces the research-work results before any behaviour changes:

| System (research work) | MedQA | PubMedQA |
|---|---|---|
| Self-MedRAG + Mistral-small | 71.30% | 75.96% |

## Demos

### Patient-style questions: the semantic gap

![First run: three patient-style questions asked live: "elephant on my chest" (withheld), "a curtain came down over one eye" (withheld; the model's unverified answer, retinal detachment, shown on request) and "blood in my poop" (a quoted source, the unverified rest hidden)](docs/media/ui-patient-queries.webp)

25 questions written the way patients describe symptoms ("it feels like an elephant is sitting
on my chest") were asked one by one through the live system (API on the Jetson, gpt-oss-120b,
MiniCheck on a rented RTX 3060), as free text with the model's own answer requested, and
compared by hand with the expected diagnosis. The first run (2026-09-27) found a semantic gap;
the second (2026-09-28) measured the changes made to close it:

| | First run | With rewrite, MedlinePlus, live PubMed |
|---|---|---|
| Answers the evidence gate showed | 2 of 25 | **13 of 25** |
| … naming the expected diagnosis (or the right urgent advice) | 0 | 8 (+2 partly) |
| Model's own answer: ✅ match · 🟡 partial · ❌ different · ⚪ declined | 7 · 3 · 3 · 12 | **15** · 5 · 4 · 1 |
| Report (every answer, statement, quote and source) | [patient_queries.md](eval/demos/patient_queries.md) | [patient_queries_v2.md](eval/demos/patient_queries_v2.md) |

- **The gap was in retrieval.** Patients' words retrieved the wrong literature ("elephant
  sitting on my chest" found papers on the *elephant trunk* aortic surgery technique), and the
  PubMedQA abstracts behind the corpus seldom state links such as symptoms to diagnosis.
- **What changed:** the question is also searched in clinical terms (`QUERY_REWRITE`: "peeing
  all the time" -> polyuria); MedlinePlus health topics (public domain, US National Library of
  Medicine) add patient-level symptom lists (`make knowledge-medlineplus`); live PubMed search
  reaches all of PubMed instead of the corpus sample (`LIVE_PUBMED`); and the agent loop now
  checks drafts with the same quoted-claim rules as the gate, so a draft without valid quotes
  is retried instead of being accepted and then withheld.
- **Shown is not the same as right.** Of the 13 shown answers, 3 are off target although every
  statement is quoted and verified: a breast lump answered with who gets a mammogram, muscle
  weakness answered with a sentence from an unrelated case report, and hand cramps with facial
  twitching called a nerve disorder (hypocalcemia expected). The gate proves each statement is
  in a source, not that the conclusion is the right diagnosis. The system is not meant for
  self-diagnosis.
- The recording above is from the first run, of three of the questions; answers vary between
  runs.

<details>
<summary>All 25 questions, second run (expected diagnosis vs the model's own answer)</summary>

| # | Patient's words | Expected | Model's own answer | Gate | Verdict |
|---|---|---|---|---|---|
| 1 | It feels like an elephant is sitting on my chest when I walk. | Stable coronary artery disease | Possible angina (chest pain on exertion) | withheld | ✅ match |
| 2 | I found a hard lump in my breast that doesn’t hurt and seems to be getting bigger. | Breast cancer (likely invasive carcinoma) | Diagnostic mammography is performed for people who have a lump or other signs or symptoms of … | **shown** | ❌ different |
| 3 | I have this sharp pain in my lower right belly that gets worse when I move. | Acute appendicitis | Appendicitis can cause sharp lower right abdominal pain that worsens over time. | **shown** | ✅ match |
| 4 | I’m always thirsty, peeing all the time, and losing weight without trying. | Diabetes mellitus (likely type 1 if acute) | Feeling very thirsty, urinating more often, and losing weight without trying are symptoms of … | **shown** | ✅ match |
| 5 | I’ve got a bad headache, stiff neck, and light really hurts my eyes. | Meningitis (likely bacterial until proven otherwise) | You should seek urgent medical evaluation as these symptoms may indicate a potentially serious … | **shown** | ✅ match |
| 6 | Suddenly I can’t see out of one eye, like a curtain came down. | Retinal detachment | A curtain‑like loss of vision in one eye is a symptom of retinal detachment. | **shown** | ✅ match |
| 7 | My hip hurts so much after I fell, and now I can’t stand on that leg. | Femoral neck fracture | You may have a hip fracture and should seek urgent medical evaluation. | withheld | 🟡 partial |
| 8 | I feel really weak, my heart feels weird, and my muscles aren’t working right. | Hyperkalemia | Weakness developed within 3-5 days of illness and myalgia was present early in the disease … | **shown** | ❌ different |
| 9 | I feel really sick, dizzy, and confused, and my family says I have a fever. | Sepsis / septic shock | Your symptoms could be consistent with sepsis. | **shown** | ✅ match |
| 10 | I suddenly felt a tearing pain in my chest that goes straight to my back. | Aortic dissection | Insufficient evidence | withheld | ⚪ declined |
| 11 | I’ve been seeing blood in my poop and my pants feel looser lately. | Colorectal cancer | Blood in the stool and looser bowel movements can be caused by several rectal conditions such … | withheld | 🟡 partial |
| 12 | I keep coughing and sometimes there’s blood, and I’ve lost weight without trying. | Lung cancer | Your symptoms could be indicative of lung cancer. | withheld | ✅ match |
| 13 | I get this bad pain on the right side under my ribs after eating greasy food. | Cholelithiasis / cholecystitis | Greasy foods can trigger gallstone attacks | **shown** | ✅ match |
| 14 | There’s a bulge in my groin that pops out when I cough or lift things. | Inguinal hernia | You likely have an inguinal (groin) hernia. | withheld | ✅ match |
| 15 | My vision gets blurry sometimes and I feel tired all the time. | Type 2 diabetes mellitus | The symptoms may be indicative of diabetes. | withheld | 🟡 partial |
| 16 | I can’t lie flat at night because I feel like I’m drowning. | Congestive heart failure | Difficulty lying flat and shortness of breath when supine are symptoms of heart failure | **shown** | ✅ match |
| 17 | My heart suddenly starts racing and feels like it’s skipping beats. | Atrial fibrillation | Your symptoms are consistent with an arrhythmia. | **shown** | 🟡 partial |
| 18 | It burns when I pee and I feel like I have to go all the time. | Urinary tract infection | You likely have a urinary tract infection. | **shown** | ✅ match |
| 19 | I’ve had a fever and chills, and my chest hurts when I breathe in. | Pneumonia | Your symptoms could be consistent with pneumonia. | withheld | ✅ match |
| 20 | I see halos around lights and my eye hurts really badly. | Acute angle-closure glaucoma | Halos around lights are a symptom of cataracts; the eye pain is not explained by the provided … | withheld | ❌ different |
| 21 | Everything looks blurry in the center, but I can still see around it. | Age-related macular degeneration | You may have macular degeneration, which causes central vision blur while peripheral vision … | withheld | ✅ match |
| 22 | My knee suddenly got swollen and painful after I twisted it. | Anterior cruciate ligament (ACL) tear | You likely have an anterior cruciate ligament (ACL) injury. | withheld | ✅ match |
| 23 | My back pain shoots down my leg like an electric shock. | Lumbar disc herniation | Your symptoms are consistent with sciatica. | **shown** | 🟡 partial |
| 24 | My hands cramp up and my face feels twitchy. | Hypocalcemia | The symptoms are consistent with a peripheral nerve disorder. | **shown** | ❌ different |
| 25 | I feel faint, sweaty, and like I might pass out after losing a lot of blood. | Hypovolemic shock | You are likely experiencing a syncopal episode caused by low blood pressure from blood loss. | withheld | ✅ match |

</details>

### A literature question and a question the corpus cannot answer

![The web UI answering two live questions: a grounded "no" with quoted sources, then "insufficient evidence" for a drug the corpus does not cover](docs/media/ui-live.webp)

A live session (recorded 2026-09-27): each question is typed into the web UI and answered by
the running system, with no replayed answers. The API ran on the Jetson, gpt-oss-120b (Groq)
wrote the answers and MiniCheck checked every statement on a rented RTX 3060, reached through
an SSH tunnel; each question cost about €0.0005. First, a literature question: both
statements are quoted from the study they come from (highlighted in its abstract) and verified,
so the answer is shown. Then a question about a drug newer than the PubMed corpus: nothing
supports an answer, so it is withheld as "insufficient evidence". Without a GPU, the UI's demo
mode replays recorded answers instead (`make web-install && make web-dev`; see [Web UI](#web-ui)).

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

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you) and Docker; for
the web UI, Node.js 20.9 or later (pnpm comes through `corepack enable`).

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

### Full stack with Docker Compose

`make up && make ingest-sample && make eval-smoke` builds and starts the whole system
(PostgreSQL + pgvector, migrations, inference service, API, web UI behind Caddy on
http://127.0.0.1:8090) with a 2,000-record corpus sample, then checks it end to end. It needs
about 1.9 GB of memory and no GPU; see [deploy/README.md](deploy/README.md). The API searches
in clinical terms and live PubMed too; run `make knowledge-medlineplus` once (download from NLM,
about 5 minutes on the Jetson's CPU) and `make up` again to add the MedlinePlus health topics.

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
- **Claim verifier for grounding: MiniCheck (`VERIFIER=minicheck`).** DeBERTa NLI stays for
  research-work parity, but it cannot tell users whether an answer is grounded: on 220
  hand-labelled statement-passage pairs its "supported" was right 79% of the time, and every
  sampled "contradiction" was an unrelated passage. `lytang/MiniCheck-RoBERTa-Large` (MIT) ranks
  support far better (average precision 0.956 vs 0.876) and held 100% precision at its default
  threshold 0.5 on 130 held-out pairs (`eval/grounding/`). It is exported to ONNX once
  (`services/inference/scripts/export_minicheck.py`, needs torch) and served on a GPU host;
  `/healthz` reports the verifier and its support threshold. Labels were made by an LLM, not
  clinicians.

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
| v3a | v2d + the verifier checks the claim of "Passage N states that X" statements | 80.0% | 78.7% | 0 | ~100% |
| v3b | v3a + a one-sentence answer claim is verified too; contradictions veto support | 77.3% | 77.3% | 0 | ~100% |
| v3c | v3a + live PubMed search (2 of 5 passages) | 78.7% | 80.0% | 0 | ~100% |
| v3d | v3a with the MiniCheck verifier in the loop | 78.7% | 74.7% | 0 | ~95% |
| v3e | v3d + claims with verbatim quotes | 77.3% | 77.3% | 0 | ~75% |
| v3f | v3e + the answer claim needs a quote too | 76.0% | 76.0% | 0 | ~95% |
| v3g | v3e + MedRAG textbooks as background knowledge | 74.7% | 76.0% | 0 | ~65% |
| v3h | v3e + clinical query rewrite, MedlinePlus, live PubMed, loop checks quotes | 80.0% | 78.7% | 0 | ~70% |
| research work | Mistral-small, same questions | 69.3% | 76.0% | – | 0% |

- With the corrected verifier the loop actually iterates, but the research-work rules then
  return the latest answer, which often backs off to "insufficient evidence" (v2a). Keeping
  the best-supported answer and preferring real answers fixes that (v2b).
- The multiple-choice constraint removes the refusals (v2c). Support stays low on MedQA
  (about 0.03): the PubMedQA-based corpus rarely backs USMLE-style reasoning, so many MedQA
  answers rely on the model's own knowledge. Low-support answers must be flagged to users.
- Differences of a few points on 150 questions are within noise (v2c and v2d differ by two
  MedQA questions); a full-set run confirms them. v2d is the configuration for the full run.
- v3a raises PubMedQA support from 0.26 to 0.39 and the share of grounded answers from 4.0% to
  10.7% at the same cost; MedQA support stays near 0 (the corpus lacks that knowledge). The API
  uses v3a.
- v3b (answer check) does not help: answer claims are rarely entailed by an abstract, so
  PubMedQA support halves (0.39 -> 0.18) and grounded answers fall from 10.7% to 5.3%, with no
  accuracy gain. It stays available (`--answer-check`) but is off.
- v3c (live PubMed, `--live-pubmed`) is within noise of v3a on accuracy and grounds a few more
  PubMedQA answers (14.7% vs 10.7%); it is available but off by default. Its real benefit,
  questions the corpus does not cover, cannot show on these questions.
- v3d-v3g are the Step 7b grounding runs; accuracy differences of 1-3 questions are within
  noise (gpt-oss answers vary between runs). What they change is grounding, below.
- v3h is the configuration made for patient-style questions (see Demos); on the golden set it
  is within noise of v3e on accuracy, at one extra LLM call per question for the rewrite
  (about EUR 0.001 per question in all).

### Step 7b: can users trust an answer?

The product must not present an answer as grounded when it is not. Re-scoring stored runs
(`eval/scripts/analyze_grounding.py`, no LLM calls) showed that the evidence statuses of
Step 7 were mostly noise, so the answer pipeline was rebuilt around an **evidence gate**:

- **Verifier:** MiniCheck instead of DeBERTa NLI (see Inference service).
- **Quoted claims** (`EVIDENCE_QUOTES`): the rationale must be factual claims, each with the
  passage and a verbatim quote. A claim is backed only if the quote really is in a retrieved
  passage, is not a statement of the study's aim or hypothesis, and the verifier finds the
  passage supports the claim.
- **Question grounding:** a statement that restates what the user stated (a vignette's
  findings) is grounded in the question and labelled so. Only the stem's declarative sentences
  count, never the answer options or the question asked ("Does X reduce Y?" asserts nothing),
  and the statement must reuse their words (80% of its content words) as well as pass the
  verifier; diagnoses and other inferences need a source.
- **Gate** (`ANSWER_POLICY=evidence_gated`): the answer is shown only when every statement is
  grounded and at least one by a source; otherwise "insufficient evidence", the studies found,
  and the model's answer only on request (`include_unverified`), labelled as unverified.

Golden 150 (evaluation runs record what the gate would show; `eval/scripts/apply_question_grounding.py`
re-applies the final rules to earlier runs):

| Run | Questions answered by the gate (MedQA · PubMedQA) | Accuracy of those | Accuracy of withheld |
|---|---|---|---|
| v3e quotes | 0% · **41%** | – · **90%** | 77% · 68% |
| v3f quotes + quoted answer claim | 0% · 1% | – · 100% | 76% · 76% |
| v3g quotes + textbooks | 16% · 47% | **58%** · 83% | 78% · 70% |
| v3h quotes + rewrite, MedlinePlus, live PubMed, loop checks quotes | 17% · **57%** | 77% · 88% | 81% · 66% |

- On literature questions (PubMedQA style) the gate answers 41% of questions, 90% of them
  correctly; the withheld ones would have been right only 68% of the time.
- **Grounded statements do not make a clinical answer correct.** With 18 medical textbooks as
  background knowledge (MedRAG, research evaluation only: no licence for products), MedQA
  answers whose every statement was quoted and verified were *less* often right (58%) than
  the withheld ones (78%): true textbook facts that do not decide between the options. A
  stricter threshold does not fix it. Textbooks are therefore not used, and the gate answers
  none of the 75 clinical-vignette questions.
- v3h (the product configuration since 2026-09-28) answers more literature questions (57%,
  88% right) and, through MedlinePlus, 17% of the clinical vignettes; those are right 77% of
  the time, a little below the withheld ones (81%), so the textbook finding holds for
  MedlinePlus too: on exam-style vignettes a verified statement does not decide between the
  options.
- Question grounding was first too loose: the web UI showed an answer option taken as a fact,
  and re-checking found questions ("Does X...?") and diagnoses passing as restatements. With
  the rules above (all runs re-gated), one statement in 450 answers is grounded in the question.
- What the product can claim: every statement it shows is quoted from a retrieved study and
  verified, or restates the user's question. It cannot claim the conclusion drawn from them is
  always right; the remaining errors are wrong inferences from true statements. The labels
  behind the verifier choice were made by an LLM; clinical review of shown answers is still
  needed before real use.

**Full evaluation of v2d** (all 1,000 MedQA and 890 PubMedQA questions; NLI on a rented GPU):

| | Agent (v2d, gpt-oss-120b) | Research work (Self-MedRAG + Mistral-small) |
|---|---|---|
| MedQA | **84.9%** | 71.3% |
| PubMedQA | 74.8% | 76.0% |
| Questions needing more than one iteration | 99.3% / 93.9% | about 1% |

MedQA improves by 13.6 points (about ±2 points of uncertainty at this size); PubMedQA is on
par (−1.1 points, within noise). MedQA support stays low (0.03), so those answers rely largely
on the model's own knowledge. PubMedQA is still open-book: the gold abstracts are retrievable.

**PubMedQA without its own abstract.** Each PubMedQA question asks about the conclusion of one
specific study, and the corpus contains that study's abstract (retrieved in the top 5 for 73 of
75 golden questions). With the question's own article excluded from retrieval
(`--leakage-free`), accuracy on the golden PubMedQA questions drops from 77.3% to 42.7%, which
is the always-"no" baseline: the agent answered "no" 73 times, with support falling from 0.26
to 0.05. PubMedQA therefore measures finding and reading the right abstract, in the research
work and here, not open-domain medical reasoning; MedQA is the better measure of that.

### API

```bash
uv run --env-file .env uvicorn medrag_api.app:app --port 8000
curl -X POST localhost:8000/v1/ask -H "Authorization: Bearer <key>" \
  -d '{"question": "Is metformin first-line for type 2 diabetes?", "answer_format": "yes_no"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | Answer as JSON |
| `POST /v1/ask/stream` | Progress events (`retrieved`, `generated`, `verified`, `refining`), then `answer` (SSE) |
| `GET /v1/runs/{id}` · `POST /v1/runs/{id}/feedback` | Stored answers and user feedback (+1 / −1) |
| `GET /healthz` · `GET /readyz` | Liveness; readiness of database, NLI, LLM key and budget |

Keys go in `API_KEYS` (bearer tokens); only `APP_ENV=local` runs without them. Each key is rate
limited, and answering stops with 503 before total LLM spend could pass 90% of `LLM_BUDGET`.
The service uses the evaluated configuration (v3e with the Step 7b evidence gate).

**Every answer says how well the literature supports it, and ungrounded answers are not
shown.** With the product settings (`VERIFIER=minicheck` on the inference service,
`EVIDENCE_QUOTES=true`, `ANSWER_POLICY=evidence_gated`; `/readyz` reports whether this
evaluated setup is active), each statement carries its support score, the verbatim quote and
the source (PubMed ID or, for background knowledge, the book) or `from_question`, and the
answer gets an evidence status:

| Evidence status | Meaning |
|---|---|
| `supported` | at least 70% of statements grounded |
| `partially_supported` | some statements grounded |
| `not_supported` | none grounded |
| `contradicted` | a passage contradicts a statement (DeBERTa NLI only; MiniCheck has no contradiction class) |
| `no_evidence` | nothing retrieved or no rationale |

The `answer` field is the model's answer only when the gate passes (every statement grounded,
at least one by a source); otherwise it is `"insufficient evidence"`, `model_answer` is withheld
unless the request sets `include_unverified`, and `note` explains why. The withheld answer's
ungrounded statements are left out too, since they would reveal it (`hidden_statements` counts
them). `ANSWER_POLICY` can be set
to `uncertain_yes_no` (the Step 7 behaviour) or `show_all` (benchmarks). Every answer lists its
citations (one per retrieved passage; `supporting_citation` links a statement to its source) and a
disclaimer. With the gate, the progress stream does not include draft answers (they are
unverified) unless `include_unverified` is set. Browsers on other origins need `CORS_ORIGINS`.

### Web UI

`apps/web` is a Next.js app exported as static files (Cloudflare Pages or any static host) that
calls the API from the browser:

```bash
cd apps/web && corepack enable && pnpm install
NEXT_PUBLIC_API_URL=mock pnpm dev          # demo mode: recorded answers, no API or GPU needed
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm dev   # the API (set CORS_ORIGINS=http://localhost:3000)
pnpm lint && pnpm typecheck && pnpm test && pnpm build   # what CI runs; build writes out/
```

It is built around the evidence gate:

- An answer is shown only when the API's gate passes; otherwise "Insufficient evidence" is a
  normal result that says why, with the studies found. The model's own answer appears only if
  the user asks for it, collapsed and marked unverified; drafts are never shown during the run.
- Each reasoning statement shows where it comes from: a verbatim quote linked to its source
  (and highlighted in the passage), "from your question", or "not found in the sources".
- Sources quoted by a statement are listed first (PubMed links; textbook passages marked as
  research-evaluation only); the other retrieved passages are collapsed.
- Progress per attempt while the agent works, a persistent disclaimer, and thumbs up/down with
  an optional comment stored with the run.

The recording at the top of this README was made with headless Chromium against the live API
(`docs/media/record-live.mjs`, `docs/media/encode_demo.py`). Demo mode replays five real golden-set answers (`eval/scripts/export_ui_fixtures.py` rebuilds
them with the API's own response code from recorded runs) and shows each example's benchmark
answer, including one whose statements are all verified but whose option is wrong.

## License

MIT. See [LICENSE](LICENSE).
