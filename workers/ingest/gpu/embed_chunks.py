"""Embed exported chunks on a GPU machine (standalone; needs torch, transformers, numpy).

Same method as medrag_inference.BgeEmbedder: BAAI/bge-small-en-v1.5 at a pinned
revision, [CLS] of the last hidden state, L2-normalised, 512-token truncation,
fp32 compute. Vectors are stored as float16 (the database keeps halfvec anyway).

    python embed_chunks.py chunks.jsonl.gz vectors.npz

Writes vectors.npz and vectors.manifest.json (counts, model, sha256 of both files).
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    with gzip.open(args.chunks, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to(device).eval()
    name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"{len(rows)} chunks on {device} ({name})")

    # Length-sorted batches keep padding small.
    lengths = [
        len(tokenizer(r["text"], truncation=True, max_length=512)["input_ids"]) for r in rows
    ]
    order = np.argsort(lengths, kind="stable")
    vectors = np.empty((len(rows), 384), dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            batch = tokenizer(
                [rows[i]["text"] for i in idx],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(device)
            cls = model(**batch).last_hidden_state[:, 0, :].float()
            vectors[idx] = torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy()
            if (start // args.batch_size) % 100 == 0:
                done = start + len(idx)
                print(f"{done}/{len(rows)} ({done / (time.time() - started):.0f}/s)", flush=True)

    arrays = {}
    counts = {}
    for profile in sorted({r["profile"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["profile"] == profile]
        arrays[f"{profile}_corpus_order"] = np.array(
            [rows[i]["corpus_order"] for i in sel], dtype=np.int64
        )
        arrays[f"{profile}_sha256"] = np.array([rows[i]["sha256"] for i in sel], dtype="S64")
        arrays[f"{profile}_vectors"] = vectors[sel].astype(np.float16)
        counts[profile] = len(sel)
    np.savez(args.out, **arrays)

    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "method": "cls, l2-normalised, max_length 512, fp32 compute, float16 storage",
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "counts": counts,
        "seconds": round(time.time() - started, 1),
        "chunks_sha256": sha256_file(args.chunks),
        "vectors_sha256": sha256_file(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
