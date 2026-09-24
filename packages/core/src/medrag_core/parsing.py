"""Parse the generator's JSON answer, with the research work's recovery rules.

`parse_generation_output` reproduces the original parser exactly, including its
quirks (it only matches JSON objects nested one level deep, and falls back to
sentence extraction otherwise). `ParsedGeneration` is the typed view the rest of
the system uses.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIDENCE = 0.7
FALLBACK_CONFIDENCE = 0.5

# Keys some models use instead of "answer".
_ANSWER_ALIASES = ("causes", "result", "response", "diagnosis", "conclusion")
# A JSON object that may contain objects nested one level deep.
_JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"[.!?]")


def _normalise(parsed: dict[str, Any]) -> dict[str, Any] | None:
    for alias in _ANSWER_ALIASES:
        if alias in parsed and "answer" not in parsed:
            parsed["answer"] = parsed.pop(alias)
            break

    if "answer" not in parsed or "rationale" not in parsed:
        return None

    rationale = parsed["rationale"]
    if isinstance(rationale, str):
        parsed["rationale"] = [rationale]
    elif isinstance(rationale, list):
        parsed["rationale"] = [r if isinstance(r, str) else str(r) for r in rationale]

    answer = parsed["answer"]
    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict):
                parts.append(" ".join(str(v) for v in item.values() if v))
            elif item:
                parts.append(str(item))
        parsed["answer"] = " ".join(parts).strip() or "See rationale."
    elif not isinstance(answer, str):
        parsed["answer"] = str(answer)

    parsed.setdefault("confidence", DEFAULT_CONFIDENCE)
    parsed.setdefault("citations", [])
    return parsed


def _fallback(text: str) -> dict[str, Any]:
    answer = ""
    answer_match = re.search(r"[Aa]nswer:\s*([^\n]+)", text)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        sentences = _SENTENCE_SPLIT.split(text)
        if sentences:
            answer = sentences[0].strip()

    rationale_match = re.search(r"[Rr]ationale:\s*\[(.*?)\]", text, re.DOTALL)
    if rationale_match:
        rationale = [s.strip().strip("\"'") for s in rationale_match.group(1).split(",")]
    else:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 10]
        rationale = sentences[1:] if len(sentences) > 1 else [text]

    return {
        "answer": answer or text[:200],
        "rationale": rationale or [text],
        "confidence": FALLBACK_CONFIDENCE,
        "citations": [],
    }


def _parse(text: str) -> tuple[dict[str, Any], bool]:
    """Return the parsed dict and whether the sentence fallback was used."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    match = _JSON_OBJECT.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            normalised = _normalise(parsed)
            if normalised is not None:
                return normalised, False

    return _fallback(text), True


def parse_generation_output(text: str) -> dict[str, Any]:
    """Parse raw model output into a dict with answer, rationale, confidence, citations.

    Extra keys from the model's JSON are kept, as in the research work.
    """
    return _parse(text)[0]


@dataclass(frozen=True)
class ParsedGeneration:
    answer: str
    rationale: list[str]
    confidence: float
    citations: list[str]
    used_fallback: bool
    """True when the output was not valid JSON with both an answer and a rationale."""
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_text(cls, text: str) -> "ParsedGeneration":
        raw, used_fallback = _parse(text)
        rationale = raw["rationale"]
        if not isinstance(rationale, list):
            rationale = [str(rationale)]
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            confidence = DEFAULT_CONFIDENCE
        citations = raw["citations"]
        if not isinstance(citations, list):
            citations = [citations]
        return cls(
            answer=raw["answer"],
            rationale=[str(r) for r in rationale],
            confidence=float(confidence),
            citations=[str(c) for c in citations],
            used_fallback=used_fallback,
            raw=raw,
        )
