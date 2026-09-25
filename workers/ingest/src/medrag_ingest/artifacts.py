"""Move the embedding stage to another machine (e.g. a rented GPU) and back.

`export_missing` writes the chunks that still lack an embedding to a gzipped
JSONL file; `workers/ingest/gpu/embed_chunks.py` turns it into an .npz of vectors;
`import_embeddings` loads those vectors. Rows are matched on
(profile, corpus_order) and checked against the chunk text hash, so database ids
never leave the database and the vector file can be reused for another database
built from the same corpus (for example on AWS).
"""

import gzip
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from medrag_db.indexes import PROFILES
from medrag_db.models import Chunk

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], npt.NDArray[np.float32]]
Vectors = dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.bytes_], npt.NDArray[np.float16]]]


@dataclass(frozen=True)
class ImportReport:
    updated: dict[str, int]
    skipped_already_embedded: dict[str, int]
    mismatched: dict[str, int]


def export_missing(
    engine: Engine, path: Path, *, profiles: Sequence[str] = PROFILES
) -> dict[str, int]:
    """Write chunks without an embedding; returns the count per profile."""
    counts = dict.fromkeys(profiles, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Session(engine) as session, gzip.open(path, "wt", encoding="utf-8") as fh:
        rows = session.execute(
            select(Chunk.profile, Chunk.corpus_order, Chunk.sha256, Chunk.text)
            .where(Chunk.profile.in_(profiles), Chunk.embedding.is_(None))
            .order_by(Chunk.profile, Chunk.corpus_order)
            .execution_options(yield_per=5000)
        )
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "profile": r.profile,
                        "corpus_order": r.corpus_order,
                        "sha256": r.sha256,
                        "text": r.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[r.profile] += 1
    logger.info("exported %s -> %s", counts, path)
    return counts


def load_vectors(path: Path) -> Vectors:
    """profile -> (corpus_order, sha256, vectors) from an embed_chunks.py .npz file."""
    out: Vectors = {}
    with np.load(path) as data:
        for profile in PROFILES:
            if f"{profile}_vectors" in data:
                out[profile] = (
                    data[f"{profile}_corpus_order"],
                    data[f"{profile}_sha256"],
                    data[f"{profile}_vectors"],
                )
    return out


def verify_sample(
    vectors: Vectors,
    export_path: Path,
    embed: EmbedFn,
    *,
    sample: int = 1000,
    min_cosine: float = 0.9999,
    seed: int = 0,
) -> float:
    """Re-embed a random sample locally; raise if any vector differs from the import file."""
    texts: dict[tuple[str, int], str] = {}
    with gzip.open(export_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            texts[(row["profile"], row["corpus_order"])] = row["text"]
    keys = list(texts)
    picked = random.Random(seed).sample(keys, min(sample, len(keys)))  # noqa: S311 - sampling, not crypto
    lookup = {
        (profile, int(order)): vecs[i]
        for profile, (orders, _, vecs) in vectors.items()
        for i, order in enumerate(orders)
    }
    local = embed([texts[k] for k in picked])
    remote = np.stack([lookup[k] for k in picked]).astype(np.float32)
    remote /= np.linalg.norm(remote, axis=1, keepdims=True)
    cos = (local * remote).sum(axis=1)
    worst = float(cos.min())
    logger.info("verify: %d vectors, cosine min %.6f mean %.6f", len(picked), worst, cos.mean())
    if worst < min_cosine:
        raise ValueError(f"imported vectors differ from local embeddings (min cosine {worst:.6f})")
    return worst


def import_embeddings(
    engine: Engine,
    vectors: Vectors,
    *,
    embed_model: str,
    embed_version: str,
    page_size: int = 2000,
) -> ImportReport:
    """Fill empty embeddings; a row is only updated if its text hash matches."""
    updated, skipped, mismatched = {}, {}, {}
    for profile, (orders, shas, vecs) in vectors.items():
        with Session(engine) as session:
            current = {
                r.corpus_order: (r.id, r.sha256, r.has_embedding)
                for r in session.execute(
                    select(
                        Chunk.corpus_order,
                        Chunk.id,
                        Chunk.sha256,
                        Chunk.embedding.is_not(None).label("has_embedding"),
                    ).where(Chunk.profile == profile)
                )
            }
        rows, skip, bad = [], 0, 0
        for order, sha, vec in zip(orders, shas, vecs, strict=True):
            found = current.get(int(order))
            sha_text = sha.decode() if isinstance(sha, bytes) else str(sha)
            if found is None or found[1] != sha_text:
                bad += 1
            elif found[2]:
                skip += 1
            else:
                rows.append({"id": found[0], "embedding": vec.astype(np.float32)})
        for start in range(0, len(rows), page_size):
            page = rows[start : start + page_size]
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Chunk),
                    [
                        {**r, "embed_model": embed_model, "embed_version": embed_version}
                        for r in page
                    ],
                )
            logger.info("%s: imported %d/%d", profile, start + len(page), len(rows))
        updated[profile], skipped[profile], mismatched[profile] = len(rows), skip, bad
    report = ImportReport(updated, skipped, mismatched)
    logger.info("import done: %s", report)
    return report
