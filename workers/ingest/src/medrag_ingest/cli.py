"""Command-line entry point.

uv run --env-file .env medrag-ingest all --limit 10000       # research-work corpus size
uv run --env-file .env medrag-ingest embed --profile parity  # resume embedding
"""

import argparse
import logging
from pathlib import Path

from medrag_db.indexes import PROFILES
from medrag_db.session import make_engine
from medrag_settings import get_settings

DEFAULT_INDEX_DIR = Path("data/indexes")


def bm25_path(index_dir: Path, profile: str) -> Path:
    return index_dir / f"bm25_{profile}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medrag-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export-chunks", help="write chunks without embeddings to a file")
    exp.add_argument("--profile", choices=PROFILES, action="append", dest="profiles")
    exp.add_argument("--out", type=Path, default=Path("data/artifacts/chunks.jsonl.gz"))
    imp = sub.add_parser("import-embeddings", help="load vectors made by gpu/embed_chunks.py")
    imp.add_argument("--vectors", type=Path, default=Path("data/artifacts/vectors.npz"))
    imp.add_argument("--chunks", type=Path, default=Path("data/artifacts/chunks.jsonl.gz"))
    imp.add_argument("--verify-sample", type=int, default=1000)
    imp.add_argument("--profile", choices=PROFILES, action="append", dest="profiles")
    for name in ("load", "embed", "index", "all"):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=PROFILES, action="append", dest="profiles")
        p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
        if name in ("load", "all"):
            p.add_argument("--limit", type=int, default=None, help="number of corpus records")
            p.add_argument("--reset", action="store_true", help="empty the corpus tables first")
    args = parser.parse_args(argv)
    profiles = tuple(args.profiles or PROFILES)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = make_engine(get_settings().database_url.get_secret_value())

    from medrag_ingest import artifacts

    if args.command == "export-chunks":
        artifacts.export_missing(engine, args.out, profiles=profiles)
        return

    from medrag_inference import BgeEmbedder
    from medrag_ingest import pipeline
    from medrag_ingest.corpus import iter_pubmedqa_sections

    embedder = BgeEmbedder()
    if args.command == "import-embeddings":
        vectors = artifacts.load_vectors(args.vectors)
        if args.verify_sample:
            artifacts.verify_sample(
                vectors, args.chunks, lambda texts: embedder.embed(texts), sample=args.verify_sample
            )
        artifacts.import_embeddings(
            engine, vectors, embed_model=embedder.model_id, embed_version=embedder.version
        )
        return
    if args.command in ("load", "all"):
        if args.reset:
            pipeline.reset_corpus(engine)
        pipeline.load_corpus(
            engine,
            iter_pubmedqa_sections(args.limit),
            profiles=profiles,
            count_tokens=embedder.count_tokens,
        )
    if args.command in ("embed", "all"):
        for profile in profiles:
            pipeline.embed_missing(engine, embedder, profile=profile)
    if args.command in ("index", "all"):
        for profile in profiles:
            pipeline.build_vector_index(engine, profile=profile)
            pipeline.build_bm25(engine, profile=profile, path=bm25_path(args.index_dir, profile))


if __name__ == "__main__":
    main()
