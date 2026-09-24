"""Capture chunker and BM25 fixtures from the research-work code (read-only, Step 4).

Same rules as capture_reference.py: run with the research-work virtualenv, nothing is
written inside the research-work project.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \\
    <research-work>/venv/bin/python eval/capture/capture_step4.py --source <research-work> --repo .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

N_BM25_DOCS = 400
N_BM25_QUERIES = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    source, repo = args.source.resolve(), args.repo.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))

    import yaml
    from rank_bm25 import BM25Okapi
    from src.corpus_loader import CorpusLoader
    from src.retrieval import RetrievalModule

    config = yaml.safe_load((source / "config_selfmedrag_mistral.yaml").read_text("utf-8"))
    loader = CorpusLoader(config)
    records = [json.loads(line) for line in open(source / "data/corpus/pubmed_abstracts.jsonl")]

    # --- chunker: every long record in the corpus, plus short and synthetic cases
    long_idx = [i for i, r in enumerate(records) if len(r["abstract"].split()) > 512][:25]
    short_idx = [i for i, r in enumerate(records[:10000]) if len(r["abstract"].split()) < 20][:10]
    normal_idx = list(range(0, 10000, 500))
    synthetic = [
        "One sentence without a final stop",
        "e.g. this abbreviation. breaks sentences. " * 60,
        ("word " * 600).strip(),
        ". ".join(f"Sentence number {i} has exactly eight words here" for i in range(120)),
        "",
    ]
    chunk_cases = []
    for i in long_idx + short_idx + normal_idx:
        text = loader._extract_text(records[i])
        chunk_cases.append({"id": f"record-{i}", "text": text, "chunks": loader._chunk_documents([text])})
    for j, text in enumerate(synthetic):
        chunk_cases.append({"id": f"synthetic-{j}", "text": text, "chunks": loader._chunk_documents([text])})

    # --- BM25: the first N_BM25_DOCS parity chunks and golden questions as queries
    corpus = loader._chunk_documents([loader._extract_text(r) for r in records[:600]])[:N_BM25_DOCS]
    tokenized = [RetrievalModule._tokenize(c) for c in corpus]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    golden = [json.loads(line) for line in open(repo / "eval/golden/golden_150.jsonl")]
    queries = [g["question"] for g in golden[: N_BM25_QUERIES - 5]] + [
        corpus[3][:200],
        "mitochondria programmed cell death lace plant",
        "zzzz unknown tokens only",
        "",
        "the of and",
    ]
    bm25_cases = []
    for q in queries:
        tokens = RetrievalModule._tokenize(q)
        scores = bm25.get_scores(tokens)
        bm25_cases.append(
            {
                "query": q,
                "scores": [float(s) for s in scores],
                "top10": [int(i) for i in np.argsort(scores)[::-1][:10]],
            }
        )

    out = repo / "packages/search/tests/fixtures/research_work"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bm25_cases.json").write_text(
        json.dumps(
            {
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "corpus": corpus,
                "average_idf": bm25.average_idf,
                "cases": bm25_cases,
            },
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    out = repo / "workers/ingest/tests/fixtures/research_work"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunker_cases.json").write_text(
        json.dumps(
            {"chunk_size_words": 512, "overlap": 50, "min_words": 20, "cases": chunk_cases},
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    print(f"chunker cases: {len(chunk_cases)}, bm25 queries: {len(bm25_cases)}")


if __name__ == "__main__":
    main()
