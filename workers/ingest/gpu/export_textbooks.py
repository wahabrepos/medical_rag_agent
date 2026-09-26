"""Export the MedRAG textbook chunks for embedding (standalone; needs huggingface_hub).

MedRAG/textbooks (Xiong et al., 2024) holds 18 English medical textbooks, as
distributed with MedQA (Jin et al., 2021), cut into snippets. It has no licence
for redistribution or products: it is used only to evaluate the research work on
MedQA-style questions and is never loaded by default (see KNOWLEDGE_DIR).

    python export_textbooks.py chunks.jsonl.gz

Writes one JSON object per snippet (id, profile, corpus_order, sha256 of the text,
book, title, text) in a fixed order and
prints the count and sha256; embed with embed_chunks.py.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

DATASET = "MedRAG/textbooks"
REVISION = "9c72838920a1323ffa867467d3f7aa7b36b0f994"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    files = sorted(
        f
        for f in list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
        if f.startswith("chunk/") and f.endswith(".jsonl")
    )
    count = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for name in files:
            path = hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION)
            book = Path(name).stem
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    text = row["content"].strip()
                    if not text:
                        continue
                    record = {
                        "id": row["id"],
                        "profile": "textbooks",
                        "corpus_order": count,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "book": book,
                        "title": row["title"],
                        "text": text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "books": len(files),
                "chunks": count,
                "sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    main()
