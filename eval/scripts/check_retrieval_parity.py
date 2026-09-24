"""Compare the new retrieval with the research work's on the 200 reference questions.

Needs the ingested corpus (medrag-ingest), the BM25 index files and the full
evaluation sets (fetch_eval_sets.py). Writes eval/reports/retrieval_parity.json.

    uv run --env-file .env python eval/scripts/check_retrieval_parity.py

Parity profile: compared chunk by chunk (text hashes), so the lists are directly
comparable. Standard profile: different chunks, so compared by PubMed article.
Passes when the parity profile's fused top-5 overlap is at least 0.8 on average.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

from medrag_db.models import Chunk, Document, Section
from medrag_db.session import make_engine, make_session_factory
from medrag_inference import BgeEmbedder
from medrag_search import BM25Index, HybridRetriever
from medrag_settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "eval/reference/retrieval_200.jsonl"
THRESHOLD = 0.8


def load_questions(data_dir: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for name in ("medqa", "pubmedqa"):
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            sys.exit(f"{path} missing: run eval/scripts/fetch_eval_sets.py first")
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            questions[row["id"]] = row
    return questions


def overlap(new: list[Any], old: list[Any]) -> float:
    return len(set(new) & set(old)) / len(set(old)) if old else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data/indexes")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--report", type=Path, default=ROOT / "eval/reports/retrieval_parity.json")
    args = ap.parse_args()

    reference = [json.loads(line) for line in REFERENCE.read_text("utf-8").splitlines()]
    questions = load_questions(args.data_dir)
    engine = make_engine(get_settings().database_url.get_secret_value())
    sessions = make_session_factory(engine)
    embedder = BgeEmbedder()

    with sessions() as session:
        sha_of = {r.id: r.sha256 for r in session.execute(select(Chunk.id, Chunk.sha256))}
        pmid_of_chunk = {
            r.id: r.pmid
            for r in session.execute(
                select(Chunk.id, Document.pmid).join(Document, Document.id == Chunk.document_id)
            )
        }
        pmid_of_line = {
            r.corpus_position: r.pmid
            for r in session.execute(
                select(Section.corpus_position, Document.pmid).join(
                    Document, Document.id == Section.document_id
                )
            )
        }

    retrievers = {
        profile: HybridRetriever(
            sessions,
            BM25Index.load(args.index_dir / f"bm25_{profile}.npz"),
            embedder.embed_query,
            profile=profile,
        )
        for profile in ("parity", "standard")
    }

    rows = []
    for ref in reference:
        question = questions[ref["id"]]["question"].strip()
        old_bm25 = [c["sha256"] for c in ref["bm25_top10"]]
        old_dense = [c["sha256"] for c in ref["dense_top10"]]
        old_fused = [c["sha256"] for c in ref["fused_top5"]]
        old_pmids = [pmid_of_line[c["line"]] for c in ref["fused_top5"]]
        gold = questions[ref["id"]].get("pubid")

        par = retrievers["parity"].search(question)
        std = retrievers["standard"].search(question)
        new_bm25 = [sha_of[i] for i in par.bm25_ids]
        new_dense = [sha_of[i] for i in par.dense_ids]
        new_fused = [sha_of[i] for i in par.fused_ids]
        std_pmids = [pmid_of_chunk[i] for i in std.fused_ids]
        rows.append(
            {
                "id": ref["id"],
                "parity": {
                    "bm25_exact": new_bm25 == old_bm25,
                    "bm25_overlap10": overlap(new_bm25, old_bm25),
                    "dense_exact": new_dense == old_dense,
                    "dense_overlap10": overlap(new_dense, old_dense),
                    "fused_exact": new_fused == old_fused,
                    "fused_overlap5": overlap(new_fused, old_fused),
                },
                "standard": {"article_overlap5": overlap(std_pmids, old_pmids)},
                "gold_article_in_top5": None
                if gold is None
                else {
                    "research_work": gold in old_pmids,
                    "parity": gold in [pmid_of_chunk[i] for i in par.fused_ids],
                    "standard": gold in std_pmids,
                },
            }
        )

    def mean(profile: str, key: str) -> float:
        return statistics.fmean(float(r[profile][key]) for r in rows)

    gold_rows = [r["gold_article_in_top5"] for r in rows if r["gold_article_in_top5"]]
    summary: dict[str, Any] = {
        "questions": len(rows),
        "parity": {k: round(mean("parity", k), 4) for k in rows[0]["parity"]},
        "standard": {"article_overlap5": round(mean("standard", "article_overlap5"), 4)},
        "pubmedqa_gold_article_in_top5": {
            k: round(statistics.fmean(float(g[k]) for g in gold_rows), 4)
            for k in ("research_work", "parity", "standard")
        },
        "threshold_fused_overlap5": THRESHOLD,
    }
    summary["passed"] = summary["parity"]["fused_overlap5"] >= THRESHOLD
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
