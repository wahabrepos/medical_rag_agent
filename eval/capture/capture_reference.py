"""Capture the research-work reference behaviour (read-only).

Runs the original Self-MedRAG code on fixed inputs and writes the results into this
repository as reference data and test fixtures:

- eval/reference/: evaluated question manifest, per-system predictions, metrics,
  retrieval results for 200 questions, and the parity settings
- eval/golden/golden_150.jsonl: the fixed pull-request evaluation subset
- packages/core/tests/fixtures/research_work/: fixtures for the pure functions that
  get ported (JSON parser, RRF, BM25 tokenizer, prompts, NLI verification, stop rules)
- eval/tests/fixtures/research_work/: answer-extraction and metric fixtures

It must run with the research-work project's own virtualenv (Python 3.10, PyTorch,
FAISS, rank_bm25). Nothing is written inside the research-work project: bytecode is
disabled, and the FAISS index and embedding model are copied to a work directory.

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \\
    <research-work>/venv/bin/python eval/capture/capture_reference.py \\
        --source <research-work> --repo . --work <scratch dir>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

GOLDEN_SEED = 42
RETRIEVAL_PER_DATASET = 100  # 200 questions in total for the Step 4 retrieval parity check
GOLDEN_PER_DATASET = 75  # the first 75 of each retrieval sample form the golden subset
CONFIG_NAME = "config_selfmedrag_mistral.yaml"

SYSTEMS = {
    "base_bm25": "BM25-only baseline (Mistral-small)",
    "base_dense": "Dense BGE baseline (Mistral-small)",
    "base_hybrid": "Hybrid RRF baseline (Mistral-small)",
    "selfmedrag_qwen_full": "Self-MedRAG + Qwen2.5-1.5B",
    "selfmedrag_qwen7b": "Self-MedRAG + Qwen2.5-7B (NF4)",
    "selfmedrag_qwen14b": "Self-MedRAG + Qwen2.5-14B (NF4)",
    "selfmedrag_mistral": "Self-MedRAG + Mistral-small",
}
PARITY_SYSTEM = "selfmedrag_mistral"
METRIC_KEYS = (
    "accuracy",
    "f1_score",
    "avg_iterations",
    "more_than_one_iter_pct",
    "avg_support_score",
    "avg_confidence",
    "latency_p50",
    "latency_p95",
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_plain(value: Any) -> Any:
    """Convert numpy scalars and containers into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(data), indent=2, ensure_ascii=False) + "\n", "utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_plain(r), ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", "utf-8")


# --------------------------------------------------------------------------- datasets


def load_eval_sets(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Rebuild the evaluated question lists exactly as the research work did.

    Returns items in evaluation order, each with its row index in the source split.
    """
    from src.dataset_loader import DatasetLoader

    loader = DatasetLoader(config)
    loaded = loader.load_data()

    medqa_df = loader._load_file_to_dataframe(loader.medqa_path)
    medqa_rows = medqa_df.sample(n=loader.sample_size, random_state=42).index.tolist()
    pubmed_df = loader._load_file_to_dataframe(loader.pubmedqa_path)
    keep = ~pubmed_df["final_decision"].astype(str).str.lower().str.strip().eq("maybe")
    pubmed_rows = pubmed_df[keep].index.tolist()
    assert len(pubmed_rows) <= loader.sample_size, "PubMedQA was not sub-sampled"

    sets: dict[str, list[dict[str, Any]]] = {}
    for name, rows, prefix in (
        ("medqa", medqa_rows, "medqa-test"),
        ("pubmedqa", pubmed_rows, "pubmedqa-labeled"),
    ):
        entries = loaded[name]
        assert len(entries) == len(rows)
        items = []
        for row, entry in zip(rows, entries):
            dtype = entry.get("dataset_type", name)
            # Same ground-truth rule as MedRAGPipeline.evaluate_dataset()
            answer = (
                entry["answer_letter"]
                if dtype == "medqa" and entry.get("answer_letter")
                else entry.get("answer", "")
            )
            items.append(
                {
                    "id": f"{prefix}-{row}",
                    "dataset": name,
                    "source_index": int(row),
                    "question": entry["question"],
                    "options": entry.get("options", []),
                    "answer": answer,
                    "answer_text": entry.get("answer_text"),
                    "dataset_type": dtype,
                }
            )
        sets[name] = items

    # Cross-check MedQA ordering against the raw file.
    raw_q = str(medqa_df.loc[medqa_rows[0], "question"]).strip()
    assert sets["medqa"][0]["question"].startswith(raw_q)
    return sets


# ------------------------------------------------------------------------ predictions


def capture_predictions(
    source: Path, sets: dict[str, list[dict[str, Any]]], repo: Path
) -> dict[str, Any]:
    from src.evaluation import Evaluation

    evaluator = Evaluation()
    metrics: dict[str, Any] = {"systems": {}, "missing": []}
    preds_by_system: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for system, label in SYSTEMS.items():
        data = json.loads((source / "results" / f"{system}.json").read_text("utf-8"))
        rows: list[dict[str, Any]] = []
        metrics["systems"][system] = {"label": label, "datasets": {}}
        preds_by_system[system] = {}
        for name in ("medqa", "pubmedqa"):
            if name not in data:
                metrics["missing"].append(f"{system}/{name}")
                continue
            block = data[name]
            preds = block["predictions"]
            items = sets[name]
            assert len(preds) == len(items), f"{system}/{name}: length mismatch"
            ground_truth = [{"answer": it["answer"]} for it in items]
            recomputed = to_plain(evaluator.evaluate(preds, ground_truth))
            stored = to_plain(block["metrics"])
            for key in METRIC_KEYS:
                assert abs(recomputed[key] - stored[key]) < 1e-9, (system, name, key)
            metrics["systems"][system]["datasets"][name] = {
                "num_samples": block["num_samples"],
                "metrics": stored,
            }
            preds_by_system[system][name] = preds
            for it, p in zip(items, preds):
                rows.append(
                    {
                        "id": it["id"],
                        "final_answer": p["final_answer"],
                        "iterations": p["iterations"],
                        "support_score": p["support_score"],
                        "confidence": p.get("confidence", 0.0),
                        "latency": p.get("latency"),
                    }
                )
        write_jsonl(repo / "eval/reference/predictions" / f"{system}.jsonl", rows)

    metrics["note"] = (
        "Metrics as stored by the research work; each was recomputed from the stored "
        "predictions with the original Evaluation class and matched to 1e-9. The hybrid "
        "baseline has no MedQA result file (71.40% was reported separately)."
    )
    write_json(repo / "eval/reference/metrics.json", metrics)
    return preds_by_system


# -------------------------------------------------------------------------- retrieval


def build_corpus(config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Load the corpus as the research work did, plus a chunk -> source-line map."""
    from src.corpus_loader import CorpusLoader

    loader = CorpusLoader(config)
    corpus = loader.load_corpus()

    source = config["corpus"]["sources"][0]
    max_docs = source.get("max_docs")
    chunk_meta: list[dict[str, Any]] = []
    with open(source["path"], encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if max_docs and line_no >= max_docs:
                break
            doc = json.loads(line)
            text = loader._extract_text(doc)
            if not text:
                continue
            for chunk in loader._chunk_documents([text]):
                chunk_meta.append(
                    {"line": line_no, "title": doc.get("title", ""), "sha256": sha256(chunk)}
                )
    assert len(chunk_meta) == len(corpus), "chunk map does not match the corpus"
    for meta, chunk in zip(chunk_meta, corpus):
        assert meta["sha256"] == sha256(chunk)
    return corpus, chunk_meta


def make_retrieval(config: dict[str, Any], corpus: list[str], work: Path, source: Path) -> Any:
    from src.retrieval import RetrievalModule

    cfg = copy.deepcopy(config)
    cache = work / "cache"
    shutil.copytree(source / "cache/faiss", cache / "faiss", dirs_exist_ok=True)
    model_dir = work / "models/bge-small-en-v1.5"
    if not model_dir.exists():
        shutil.copytree(source / "models/bge-small-en-v1.5", model_dir)
    cfg["system"]["cache_dir"] = str(cache)
    cfg["retrieval"]["dense"]["cache_dir"] = str(model_dir)
    before = sorted(p.name for p in (cache / "faiss").iterdir())
    module = RetrievalModule(cfg, corpus=corpus)
    after = sorted(p.name for p in (cache / "faiss").iterdir())
    assert before == after, "FAISS cache key mismatch: the index was rebuilt"
    return module


def pick_samples(sets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(GOLDEN_SEED)
    picked = {}
    for name in ("medqa", "pubmedqa"):
        idx = rng.sample(range(len(sets[name])), RETRIEVAL_PER_DATASET)
        picked[name] = [sets[name][i] for i in idx]
    return picked


def capture_retrieval(
    retrieval: Any,
    corpus: list[str],
    chunk_meta: list[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    top_k: int,
    repo: Path,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    first_index: dict[str, int] = {}
    for i, chunk in enumerate(corpus):
        first_index.setdefault(chunk, i)

    def describe(docs: list[str]) -> list[dict[str, Any]]:
        out = []
        for d in docs:
            i = first_index[d]
            out.append({"chunk": i, "line": chunk_meta[i]["line"], "sha256": chunk_meta[i]["sha256"]})
        return out

    rows = []
    fused_texts: dict[str, list[str]] = {}
    rrf_cases = []
    for name in ("medqa", "pubmedqa"):
        for item in samples[name]:
            query = item["question"].strip()  # first-iteration query in Trainer.run()
            res = retrieval.retrieve(query)
            fused_full = retrieval.fuse_results(res["bm25"], res["dense"])
            fused = fused_full[:top_k]
            fused_texts[item["id"]] = fused
            rows.append(
                {
                    "id": item["id"],
                    "query_sha256": sha256(query),
                    "bm25_top10": describe(res["bm25"]),
                    "dense_top10": describe(res["dense"]),
                    "fused_top5": describe(fused),
                }
            )
            as_ids = lambda docs: [f"c{first_index[d]}" for d in docs]  # noqa: E731
            rrf_cases.append(
                {
                    "id": item["id"],
                    "bm25": as_ids(res["bm25"]),
                    "dense": as_ids(res["dense"]),
                    "expected": as_ids(fused_full),
                }
            )
    write_jsonl(repo / "eval/reference/retrieval_200.jsonl", rows)

    # Synthetic edge cases, answered by the original fuse_results()
    edge = [
        {"id": "edge-empty", "bm25": [], "dense": []},
        {"id": "edge-bm25-only", "bm25": ["a", "b", "c"], "dense": []},
        {"id": "edge-identical", "bm25": ["a", "b", "c"], "dense": ["a", "b", "c"]},
        {"id": "edge-reversed", "bm25": ["a", "b", "c"], "dense": ["c", "b", "a"]},
        {"id": "edge-tie", "bm25": ["a", "b"], "dense": ["b", "a"]},
        {"id": "edge-duplicate-in-list", "bm25": ["a", "a", "b"], "dense": ["b"]},
    ]
    for case in edge:
        case["expected"] = retrieval.fuse_results(case["bm25"], case["dense"])
        rrf_cases.append(case)
    write_json(
        repo / "packages/core/tests/fixtures/research_work/rrf_cases.json",
        {"rrf_k": retrieval.rrf_k, "cases": rrf_cases},
    )

    # BM25 tokenizer on every sampled query plus some corpus chunks
    texts = [it["question"] for n in samples for it in samples[n]]
    texts += [corpus[i] for i in range(0, len(corpus), max(1, len(corpus) // 20))][:20]
    texts += ["COVID-19 and TNF-alpha in HbA1c-positive patients; IL-6 -- beta-blocker (n=12)."]
    tok = [{"text": t, "tokens": retrieval._tokenize(t)} for t in texts]
    write_json(repo / "packages/core/tests/fixtures/research_work/bm25_tokenizer_cases.json", tok)
    return fused_texts, rows


# ------------------------------------------------------------------ generator helpers


def make_generator(config: dict[str, Any]) -> Any:
    from src.model import GeneratorModule

    gen = GeneratorModule.__new__(GeneratorModule)
    gcfg = config["generator"]
    gen.model_type = gcfg.get("model_type", "causal")
    gen.system_prompt = gcfg.get("system_prompt", gen._default_system_prompt())
    gen.user_prompt_template = gcfg.get("user_prompt_template", gen._default_user_prompt())
    return gen


def api_messages(gen: Any, prompt: str) -> dict[str, str]:
    """Same system/user split as GeneratorModule._generate_via_api()."""
    sys_text = gen.system_prompt.strip()
    separator = gen.system_prompt + "\n\n"
    if prompt.startswith(separator):
        user_text = prompt[len(separator) :].strip()
    elif prompt.startswith(sys_text):
        user_text = prompt[len(sys_text) :].strip()
    else:
        user_text = prompt.strip()
    return {"system": sys_text, "user": user_text}


def capture_prompts(
    gen: Any,
    samples: dict[str, list[dict[str, Any]]],
    fused: dict[str, list[str]],
    repo: Path,
) -> None:
    cases = []
    history = [
        {"query": "first query", "answer": "B", "support_score": 0.4},
        {"query": "second query", "answer": "C", "support_score": 0.55},
        {"query": "third query", "answer": "C", "support_score": 0.6},
    ]
    for name in ("medqa", "pubmedqa"):
        for item in samples[name][:3]:
            ctx = fused[item["id"]]
            for label, hist in (("no-history", []), ("with-history", history)):
                prompt = gen._construct_prompt(
                    item["question"].strip(), ctx, hist, dataset_type=item["dataset_type"]
                )
                cases.append(
                    {
                        "id": f"{item['id']}/{label}",
                        "query": item["question"].strip(),
                        "context": ctx,
                        "history": hist,
                        "dataset_type": item["dataset_type"],
                        "prompt": prompt,
                        "messages": api_messages(gen, prompt),
                    }
                )
    write_json(
        repo / "packages/core/tests/fixtures/research_work/prompt_cases.json",
        {
            "system_prompt": gen.system_prompt,
            "user_prompt_template": gen.user_prompt_template,
            "cases": cases,
        },
    )


def capture_parser(gen: Any, preds: dict[str, list[dict[str, Any]]], repo: Path) -> None:
    samples = [p for p in preds["medqa"][:3]] + [p for p in preds["pubmedqa"][:3]]
    inputs: list[tuple[str, str]] = []
    for i, p in enumerate(samples):
        obj = {
            "answer": p["final_answer"],
            "rationale": p["rationale"],
            "confidence": p.get("confidence", 0.8),
            "citations": ["Passage 1"],
        }
        body = json.dumps(obj, indent=2, ensure_ascii=False)
        inputs += [
            (f"real-{i}-plain", body),
            (f"real-{i}-fenced", f"```json\n{body}\n```"),
            (f"real-{i}-prose-around", f"Here is my answer:\n{body}\nHope this helps."),
        ]
    inputs += [
        ("alt-key-diagnosis", '{"diagnosis": "Sepsis", "rationale": ["Fever and hypotension"]}'),
        ("alt-key-result", '{"result": "yes", "rationale": "Supported by passage 2"}'),
        ("answer-list-of-dicts", '{"answer": [{"drug": "Metformin", "dose": "500 mg"}], "rationale": ["r"]}'),
        ("answer-list-mixed", '{"answer": ["A", "", "B"], "rationale": ["r"]}'),
        ("answer-empty-list", '{"answer": [], "rationale": ["r"]}'),
        ("answer-number", '{"answer": 3, "rationale": ["r"], "confidence": 0.9}'),
        ("rationale-numbers", '{"answer": "A", "rationale": [1, 2.5, "three"]}'),
        ("missing-optional", '{"answer": "no", "rationale": ["Not supported"]}'),
        ("nested-one-level", '{"answer": "B", "rationale": ["x"], "meta": {"k": 1}}'),
        ("nested-two-levels", '{"answer": "B", "rationale": ["x"], "meta": {"k": {"z": 1}}}'),
        ("missing-rationale", '{"answer": "B", "confidence": 0.9}'),
        ("trailing-comma", '{"answer": "C", "rationale": ["a",],}\nAnswer: C'),
        ("two-objects", '{"answer": "A", "rationale": ["first"]} {"answer": "B", "rationale": ["second"]}'),
        ("prose-answer-line", "Answer: D\nRationale: [\"because of X\", \"and Y\"]"),
        ("prose-sentences", "The most likely diagnosis is pneumonia. The chest X-ray shows consolidation. Fever supports it."),
        ("prose-short", "yes"),
        ("empty", ""),
        ("whitespace", "   \n  "),
        ("unicode", '{"answer": "β-blocker ≥ 50 mg", "rationale": ["Dosis für Erwachsene"]}'),
    ]
    cases = [{"id": cid, "input": text, "expected": gen._parse_json_output(text)} for cid, text in inputs]
    write_json(repo / "packages/core/tests/fixtures/research_work/parser_cases.json", cases)


# ------------------------------------------------------------------ verification (NLI)


def capture_verification(repo: Path) -> None:
    from src.model import SelfReflectiveModule

    cases_in = [
        ("all-supported", ["s1", "s2"], ["p1", "p2"], [0.9, 0.1, 0.2, 0.8]),
        ("one-unsupported", ["s1", "s2", "s3"], ["p1", "p2"], [0.9, 0.1, 0.3, 0.2, 0.1, 0.75]),
        ("boundary-equal", ["s1"], ["p1", "p2"], [0.7, 0.2]),
        ("boundary-below", ["s1"], ["p1"], [0.6999]),
        ("none-supported", ["s1", "s2"], ["p1"], [0.1, 0.2]),
        ("empty-rationale", [], ["p1"], []),
        ("no-passages", ["s1", "s2"], [], []),
        ("batch-failure-zeros", ["s1", "s2"], ["p1", "p2"], [0.0, 0.0, 0.0, 0.0]),
    ]
    out = []
    for cid, rationale, context, probs in cases_in:
        mod = SelfReflectiveModule.__new__(SelfReflectiveModule)
        mod.verification_threshold = 0.7
        mod.nli_batch_size = 16
        seen: list[list[str]] = []

        def fake_batch(pairs: list[tuple[str, str]], _p: list[float] = probs) -> list[float]:
            seen.extend([list(pair) for pair in pairs])
            return list(_p)

        mod._run_nli_batch = fake_batch
        score, unsupported = mod.verify_and_extract(rationale, context)
        out.append(
            {
                "id": cid,
                "rationale": rationale,
                "context": context,
                "entailment_probs": probs,
                "expected_pairs": seen,
                "expected_support_score": score,
                "expected_unsupported": unsupported,
            }
        )
    write_json(
        repo / "packages/core/tests/fixtures/research_work/verification_cases.json",
        {"verification_threshold": 0.7, "cases": out},
    )


# -------------------------------------------------------------------- stop rules loop


class _Retrieval:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> dict[str, list[str]]:
        self.queries.append(query)
        return {"bm25": ["b1", "b2"], "dense": ["d1", "b1"]}

    def fuse_results(self, bm25: list[str], dense: list[str]) -> list[str]:
        return list(dict.fromkeys(bm25 + dense))


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, query, context, history, dataset_type=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        n = self.calls
        return f"answer-{n}", [f"stmt-{n}-a", f"stmt-{n}-b"], 0.5 + n / 10, [f"cite-{n}"]


class _Reflector:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls = 0

    def verify_and_extract(self, rationale, context):  # type: ignore[no-untyped-def]
        step = self.script[self.calls]
        self.calls += 1
        if step == "raise":
            raise RuntimeError("scripted failure")
        score, unsupported = step
        return score, list(unsupported)


def capture_stop_rules(config: dict[str, Any], repo: Path) -> None:
    from src.trainer import Trainer

    un = ["unsupported claim one", "unsupported claim two", "claim three", "claim four"]
    scripts = [
        ("meets-threshold-first", "structured", [(0.8, [])]),
        ("meets-threshold-second", "structured", [(0.5, un[:2]), (0.75, [])]),
        ("threshold-exactly-equal", "structured", [(0.6, un[:1]), (0.7, [])]),
        ("early-stop-small-gain", "structured", [(0.5, un[:2]), (0.52, un[:1])]),
        ("early-stop-drop-third", "structured", [(0.3, un), (0.4, un), (0.35, un)]),
        ("exhaust-best-of-history", "structured", [(0.4, un), (0.5, un), (0.6, un)]),
        ("no-unsupported-keeps-query", "structured", [(0.5, []), (0.6, []), (0.69, [])]),
        ("error-first-iteration", "structured", ["raise"]),
        ("error-second-iteration", "structured", [(0.5, un[:1]), "raise"]),
        ("strategy-decomposition", "decomposition", [(0.4, un), (0.5, un), (0.6, un)]),
        ("strategy-concatenation", "concatenation", [(0.4, un), (0.5, un), (0.6, un)]),
    ]
    out = []
    for cid, strategy, script in scripts:
        cfg = copy.deepcopy(config)
        cfg["iteration"]["refinement_strategy"] = strategy
        retrieval, generator, reflector = _Retrieval(), _Generator(), _Reflector(script)
        trainer = Trainer(retrieval, generator, reflector, cfg)
        answer, rationale, iterations, support, history = trainer.run("  What is the question?  ")
        out.append(
            {
                "id": cid,
                "refinement_strategy": strategy,
                "reflector_script": [s if s == "raise" else list(s) for s in script],
                "expected": {
                    "answer": answer,
                    "rationale": rationale,
                    "iterations": iterations,
                    "support_score": support,
                    "retrieval_queries": retrieval.queries,
                    "history": [
                        {k: v for k, v in h.items() if k != "context"} for h in history
                    ],
                },
            }
        )
    it = config["iteration"]
    write_json(
        repo / "packages/core/tests/fixtures/research_work/stop_rule_cases.json",
        {
            "settings": {
                "max_iterations": it["max_iterations"],
                "early_stopping": it["early_stopping"],
                "min_improvement": it["min_improvement"],
                "rationale_score_threshold": config["self_reflection"]["rationale_score_threshold"],
                "top_k": config["retrieval"]["top_k"],
            },
            "fakes": {
                "generator": "call n returns answer-n, [stmt-n-a, stmt-n-b], confidence 0.5+n/10, [cite-n]",
                "retrieval": "bm25 [b1, b2], dense [d1, b1], fused = order-preserving union",
            },
            "cases": out,
        },
    )


# ------------------------------------------------------------------ evaluation checks


def capture_evaluation(
    sets: dict[str, list[dict[str, Any]]],
    preds_by_system: dict[str, dict[str, list[dict[str, Any]]]],
    golden_ids: set[str],
    repo: Path,
) -> None:
    from src.evaluation import Evaluation

    ev = Evaluation()
    letters: dict[str, str] = {}
    binary: dict[str, str] = {}
    for per_ds in preds_by_system.values():
        for p in per_ds.get("medqa", []):
            letters.setdefault(p["final_answer"], ev.extract_option_letter(p["final_answer"]))
        for p in per_ds.get("pubmedqa", []):
            binary.setdefault(p["final_answer"], ev.extract_binary_answer(p["final_answer"]))
    extra_letters = ["A", "b.", "(C)", "[d]", "The answer is E", "Option: b", "none", "", "A)x", "xyz a"]
    extra_binary = ["Yes", "no.", "YES, it does", "Maybe", "", "not sure", "yes and no", "know"]
    for t in extra_letters:
        letters.setdefault(t, ev.extract_option_letter(t))
    for t in extra_binary:
        binary.setdefault(t, ev.extract_binary_answer(t))
    write_json(
        repo / "eval/tests/fixtures/research_work/extract_cases.json",
        {
            "option_letter": [{"input": k, "expected": v} for k, v in letters.items()],
            "binary": [{"input": k, "expected": v} for k, v in binary.items()],
        },
    )

    subset_metrics = []
    for system, per_ds in preds_by_system.items():
        for name, preds in per_ds.items():
            pairs = [
                (p, {"answer": it["answer"]})
                for it, p in zip(sets[name], preds)
                if it["id"] in golden_ids
            ]
            got = ev.evaluate([p for p, _ in pairs], [g for _, g in pairs])
            subset_metrics.append(
                {"system": system, "dataset": name, "n": len(pairs), "expected": got}
            )
    write_json(repo / "eval/tests/fixtures/research_work/golden_metrics_cases.json", subset_metrics)


# ------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    source, repo, work = args.source.resolve(), args.repo.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    os.chdir(source)  # the research-work code resolves data paths relative to its root
    sys.path.insert(0, str(source))
    import yaml

    config = yaml.safe_load((source / CONFIG_NAME).read_text("utf-8"))

    print("datasets ...", flush=True)
    sets = load_eval_sets(config)
    manifest = {
        name: [
            {
                "id": it["id"],
                "source_index": it["source_index"],
                "question_sha256": sha256(it["question"]),
                "answer": it["answer"],
            }
            for it in items
        ]
        for name, items in sets.items()
    }
    manifest_meta = {
        "medqa": "openlifescienceai/medqa test split (1,273 rows) -> 1,000 sampled with "
        "pandas DataFrame.sample(n=1000, random_state=42); question formatted with "
        "lettered options and a single-letter instruction",
        "pubmedqa": "qiaojin/PubMedQA pqa_labeled (1,000 rows) -> 'maybe' removed -> 890; "
        "question text only (no abstract)",
    }
    write_json(repo / "eval/reference/eval_sets_manifest.json", {"sources": manifest_meta, **manifest})

    print("predictions + metrics ...", flush=True)
    preds_by_system = capture_predictions(source, sets, repo)

    print("corpus + retrieval ...", flush=True)
    corpus, chunk_meta = build_corpus(config)
    retrieval = make_retrieval(config, corpus, work, source)
    samples = pick_samples(sets)
    fused, _ = capture_retrieval(
        retrieval, corpus, chunk_meta, samples, config["retrieval"]["top_k"], repo
    )

    golden = {n: samples[n][:GOLDEN_PER_DATASET] for n in samples}
    golden_ids = {it["id"] for n in golden for it in golden[n]}
    ref_rows = {
        s: {
            it["id"]: p
            for n, preds in per.items()
            for it, p in zip(sets[n], preds)
            if it["id"] in golden_ids
        }
        for s, per in preds_by_system.items()
    }
    golden_rows = []
    for name in ("medqa", "pubmedqa"):
        for it in golden[name]:
            ref = {}
            for system in SYSTEMS:
                p = ref_rows[system].get(it["id"])
                if p is None:
                    continue
                entry = {
                    "final_answer": p["final_answer"],
                    "iterations": p["iterations"],
                    "support_score": p["support_score"],
                    "confidence": p.get("confidence", 0.0),
                }
                if system == PARITY_SYSTEM:
                    entry["rationale"] = p["rationale"]
                ref[system] = entry
            golden_rows.append(
                {
                    "id": it["id"],
                    "dataset": name,
                    "question": it["question"],
                    "options": it["options"],
                    "answer": it["answer"],
                    "answer_text": it["answer_text"],
                    "reference": ref,
                }
            )
    write_jsonl(repo / "eval/golden/golden_150.jsonl", golden_rows)

    print("fixtures ...", flush=True)
    gen = make_generator(config)
    capture_prompts(gen, samples, fused, repo)
    capture_parser(gen, preds_by_system[PARITY_SYSTEM], repo)
    capture_verification(repo)
    capture_stop_rules(config, repo)
    capture_evaluation(sets, preds_by_system, golden_ids, repo)

    gcfg, it = config["generator"], config["iteration"]
    write_json(
        repo / "eval/reference/parity_settings.json",
        {
            "source_config": CONFIG_NAME,
            "generator": {
                "provider": gcfg["api_fallback"]["provider"],
                "model": gcfg["api_fallback"]["model"],
                "max_tokens": gcfg["max_new_tokens"],
                "temperature": None,
                "temperature_note": "Not sent in the API call, so the provider default "
                "applied (the config's 0.1 only affected the local model).",
                "system_prompt": gcfg["system_prompt"],
                "user_prompt_template": gen.user_prompt_template,
                "pubmedqa_binary_constraint": True,
            },
            "retrieval": {
                "corpus": "first 10,000 records of pubmed_abstracts.jsonl; abstract text "
                "only (title not included); sentence chunks of <=512 words, min 20 words",
                "num_chunks": len(corpus),
                "bm25": config["retrieval"]["bm25"],
                "dense_model": config["retrieval"]["dense"]["model_name"],
                "dense_pooling": "CLS token, L2-normalised, max_length 512",
                "per_retriever_top_k": 10,
                "rrf_k": config["retrieval"]["rrf"]["k"],
                "rrf_weights_applied": False,
                "top_k": config["retrieval"]["top_k"],
            },
            "verification": {
                "nli_model": "cross-encoder/nli-deberta-v3-base",
                "entailment_label_index": 2,
                "verification_threshold": config["self_reflection"]["verification_threshold"],
                "max_length": 512,
            },
            "loop": {
                "max_iterations": it["max_iterations"],
                "early_stopping": it["early_stopping"],
                "min_improvement": it["min_improvement"],
                "max_time_seconds": it["max_time_seconds"],
                "refinement_strategy": it["refinement_strategy"],
                "rationale_score_threshold": config["self_reflection"]["rationale_score_threshold"],
            },
            "targets": {"medqa_accuracy": 0.713, "pubmedqa_accuracy": 0.7596, "tolerance_pp": 2.0},
        },
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
