"""Build a knowledge index from MedlinePlus health topics (NLM).

MedlinePlus health topic summaries are written by the US National Library of Medicine and
are in the public domain; reuse needs the attribution "Courtesy of MedlinePlus from the
National Library of Medicine". Only these summaries are used: the A.D.A.M. encyclopedia and
drug monographs that MedlinePlus also shows are copyrighted and are not in this file.

They state the plain symptom-to-condition knowledge that research abstracts rarely do
("frequent urination, feeling thirsty ... are symptoms of diabetes"), in patient wording.

    curl -LO https://medlineplus.gov/xml/mplus_topics_compressed_<YYYY-MM-DD>.zip && unzip ...
    uv run python -m medrag_ingest.medlineplus mplus_topics_<YYYY-MM-DD>.xml \
        --out data/knowledge/medlineplus

Writes the files medrag_search.knowledge.KnowledgeIndex loads (chunks.jsonl.gz,
vectors.npz, vectors.manifest.json); BM25 is built on first load.
"""

import argparse
import gzip
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import numpy as np
from defusedxml import ElementTree

PROFILE = "medlineplus"
ATTRIBUTION = "Courtesy of MedlinePlus from the National Library of Medicine"
MAX_WORDS = 150


HEADING, TEXT, ITEM = "heading", "text", "item"


class _Blocks(HTMLParser):
    """Text blocks of a summary's HTML, each marked as heading, paragraph or list item."""

    KINDS: ClassVar[dict[str, str]] = {
        "p": TEXT,
        "li": ITEM,
        "h2": HEADING,
        "h3": HEADING,
        "h4": HEADING,
    }

    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[tuple[str, str]] = []  # (kind, text)
        self._current: list[str] = []
        self._kind = TEXT

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self.KINDS:
            self._flush()
            self._kind = self.KINDS[tag]

    def handle_endtag(self, tag: str) -> None:
        if tag in self.KINDS:
            self._flush()
            self._kind = TEXT

    def handle_data(self, data: str) -> None:
        self._current.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._current).split())
        if text:
            self.blocks.append((self._kind, text))
        self._current = []

    def close(self) -> None:
        super().close()
        self._flush()


# Credit lines at the end of summaries ("NIH: National Institute of ..."), not content.
_CREDIT = re.compile(r"^[A-Z][A-Za-z&.]{1,15}: [A-Z][^.]*$")


def summary_blocks(html: str) -> list[tuple[str, str]]:
    parser = _Blocks()
    parser.feed(html)
    parser.close()
    return [(k, t) for k, t in parser.blocks if not (k == TEXT and _CREDIT.match(t))]


def chunks_of(blocks: list[tuple[str, str]], max_words: int = MAX_WORDS) -> list[str]:
    """One chunk per section (a heading and the blocks under it); a long section is split
    and each piece starts with its heading, so no piece loses what it is about.

    A list ends its chunk and stays with the sentence that introduces it ("Symptoms may
    include:"), repeated when a long list is split: short passages that state one list
    whole are what the claim verifier checks reliably, and each item keeps its subject."""
    sections: list[tuple[str, list[tuple[str, str]]]] = [("", [])]
    for kind, text in blocks:
        if kind == HEADING:
            sections.append((text, []))
        else:
            sections[-1][1].append((kind, text if re.search(r"[.!?:]$", text) else text + "."))
    return [
        chunk for heading, body in sections for chunk in _section_chunks(heading, body, max_words)
    ]


def _section_chunks(heading: str, body: list[tuple[str, str]], max_words: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    words = 0
    lead = ""  # the sentence introducing the list being read
    previous = TEXT
    for kind, text in body:
        n = len(text.split())
        list_ended = kind == TEXT and previous == ITEM
        if current and (list_ended or words + n > max_words):
            out.append(" ".join([heading, *current]).strip())
            current = [lead] if kind == ITEM and lead else []
            words = len(lead.split()) if current else 0
        if kind == TEXT:
            lead = text if text.endswith(":") else ""
        current.append(text)
        words += n
        previous = kind
    if current:
        out.append(" ".join([heading, *current]).strip())
    return out


def topics(xml_path: Path) -> list[dict[str, object]]:
    """English health topics with their summary, names and page URL."""
    root = ElementTree.parse(xml_path).getroot()
    found = []
    for topic in root.iter("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.findtext("full-summary") or ""
        found.append(
            {
                "id": topic.get("id"),
                "title": topic.get("title", "").strip(),
                "url": topic.get("url"),
                "also_called": [a.text.strip() for a in topic.iter("also-called") if a.text],
                "blocks": summary_blocks(summary),
            }
        )
    return found


def records(xml_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topic in topics(xml_path):
        title = str(topic["title"])
        pieces = chunks_of(topic["blocks"])  # type: ignore[arg-type]
        for i, piece in enumerate(pieces):
            text = f"{title}: {piece}"
            if i == 0 and topic["also_called"]:
                text = f"{title} (also called {', '.join(topic['also_called'])}): {piece}"  # type: ignore[arg-type]
            rows.append(
                {
                    "id": f"medlineplus-{topic['id']}-{i}",
                    "profile": PROFILE,
                    "corpus_order": len(rows),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "book": "MedlinePlus",
                    "source": "medlineplus",
                    "title": title,
                    "url": topic["url"],
                    "text": text,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/knowledge/medlineplus"))
    args = ap.parse_args()

    from medrag_inference import BgeEmbedder

    rows = records(args.xml)
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = args.out / "chunks.jsonl.gz"
    with gzip.open(chunks, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    embedder = BgeEmbedder()
    vectors = embedder.embed([str(r["text"]) for r in rows], batch_size=32)
    out = args.out / "vectors.npz"
    arrays: dict[str, np.ndarray] = {
        f"{PROFILE}_corpus_order": np.arange(len(rows), dtype=np.int64),
        f"{PROFILE}_vectors": vectors.astype(np.float16),
    }
    np.savez(out, **arrays)  # type: ignore[arg-type]
    (args.out / "bm25.npz").unlink(missing_ok=True)  # rebuilt for the new chunks
    manifest = {
        "source": args.xml.name,
        "attribution": ATTRIBUTION,
        "topics": len({str(r["title"]) for r in rows}),
        "chunks": len(rows),
        "model": embedder.version,
        "chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest(),
        "vectors_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (args.out / "vectors.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
