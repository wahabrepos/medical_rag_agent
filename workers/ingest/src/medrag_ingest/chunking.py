"""Chunkers for the two retrieval profiles.

- `chunk_research_work`: the research work's word-based sentence chunker, ported
  exactly (it splits on ". ", packs up to 512 words, repeats the last sentence as
  overlap and drops chunks under 20 words). Used by the "parity" profile.
- `chunk_by_tokens`: token-aware chunking with a real sentence splitter, for the
  "standard" profile.
"""

import re
from collections.abc import Callable

# Abbreviations that end with a period but do not end a sentence. Units ("mg.") and
# "etc." are left out: in abstracts they usually do end the sentence.
_ABBREVIATIONS = frozenset(
    {
        "e.g.", "i.e.", "vs.", "al.", "fig.", "figs.", "no.", "nos.", "dr.", "approx.",
        "ca.", "cf.", "resp.", "vol.", "ref.",
    }
)  # fmt: skip
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def chunk_research_work(
    text: str, *, chunk_size: int = 512, overlap: int = 50, min_words: int = 20
) -> list[str]:
    """Research-work chunking of one document, reproduced exactly."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        n = len(sentence.split())
        if length + n > chunk_size:
            if current:
                chunk = ". ".join(current) + "."
                if len(chunk.split()) >= min_words:
                    chunks.append(chunk)
            # The research work keeps overlap // 50 sentences (0 would keep all of them).
            current = [*current[-(overlap // 50) :], sentence]
            length = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            length += n
    if current:
        chunk = ". ".join(current) + "."
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by a capital or digit, keeping
    common abbreviations ("e.g.", "et al.", "Fig.") inside their sentence."""
    pieces = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].split()[-1].lower() in _ABBREVIATIONS:
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def chunk_by_tokens(
    text: str,
    count_tokens: Callable[[str], int],
    *,
    max_tokens: int = 350,
    overlap_sentences: int = 1,
) -> list[str]:
    """Pack whole sentences into chunks of at most `max_tokens` tokens.

    Consecutive chunks share `overlap_sentences` sentences. A sentence longer than
    the limit is split into word windows that fit.
    """
    units: list[str] = []
    for sentence in split_sentences(text):
        if count_tokens(sentence) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(_split_long(sentence, count_tokens, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join([*current, unit])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            kept = current[-overlap_sentences:] if overlap_sentences else []
            if kept and count_tokens(" ".join([*kept, unit])) > max_tokens:
                kept = []
            current = [*kept, unit]
        else:
            current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_long(sentence: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if current and count_tokens(" ".join([*current, word])) > max_tokens:
            windows.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        windows.append(" ".join(current))
    return windows
