"""The retrieval corpus: PubMedQA abstracts, one record per labelled abstract section.

This reproduces the research-work corpus exactly (203,429 records): PubMedQA
`pqa_labeled` followed by `pqa_unlabeled`, every context section longer than 30
characters, in dataset order. Each record now also carries its PubMed ID and
section label.

Note: `pqa_labeled` contains the abstracts of the PubMedQA evaluation questions,
so those gold abstracts are retrievable, as they were in the research work.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

PUBMEDQA_REPO = "qiaojin/PubMedQA"
SUBSETS = ("pqa_labeled", "pqa_unlabeled")
MIN_SECTION_CHARS = 31  # the research-work corpus kept sections longer than 30 characters


@dataclass(frozen=True)
class SectionRecord:
    corpus_position: int
    pmid: int
    subset: str
    title: str
    section_position: int
    label: str | None
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def iter_pubmedqa_sections(limit: int | None = None) -> Iterator[SectionRecord]:
    """Corpus records in research-work order; `limit` stops after that many records."""
    from datasets import load_dataset

    position = 0
    for subset in SUBSETS:
        for item in load_dataset(PUBMEDQA_REPO, subset, split="train"):
            context = item["context"]
            labels = context.get("labels") or []
            meta = {"subset": subset, "meshes": context.get("meshes") or []}
            for i, text in enumerate(context["contexts"]):
                if len(text) < MIN_SECTION_CHARS:
                    continue
                if limit is not None and position >= limit:
                    return
                yield SectionRecord(
                    corpus_position=position,
                    pmid=int(item["pubid"]),
                    subset=subset,
                    title=str(item["question"]),
                    section_position=i,
                    label=labels[i] if i < len(labels) else None,
                    text=text,
                    meta=meta,
                )
                position += 1
