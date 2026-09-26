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
# Lenient field extraction for almost-valid JSON (not in the research work).
_JSON_STRING = r'"((?:[^"\\]|\\.)*)"'
_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*' + _JSON_STRING, re.DOTALL)
_RATIONALE_FIELD = re.compile(r'"rationale"\s*:\s*\[(.*?)\]', re.DOTALL)
_CONFIDENCE_FIELD = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)(?=\s*[,}\n])')


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


def _lenient(text: str) -> dict[str, Any] | None:
    """Read the answer, rationale and confidence fields out of invalid JSON.

    Models occasionally write almost-valid JSON (for example `"confidence": 0. nine`);
    the research-work fallback then keeps the raw text as the answer.
    """
    answer = _ANSWER_FIELD.search(text)
    if answer is None:
        return None
    rationale_match = _RATIONALE_FIELD.search(text)
    rationale = (
        [_unescape(m) for m in re.findall(_JSON_STRING, rationale_match.group(1))]
        if rationale_match
        else []
    )
    confidence = _CONFIDENCE_FIELD.search(text)
    return {
        "answer": _unescape(answer.group(1)).strip(),
        "rationale": rationale or [_unescape(answer.group(1))],
        "confidence": float(confidence.group(1)) if confidence else DEFAULT_CONFIDENCE,
        "citations": [],
    }


def _unescape(value: str) -> str:
    try:
        decoded: str = json.loads(f'"{value}"')
        return decoded
    except json.JSONDecodeError:
        return value


def _parse(text: str, *, lenient: bool = False) -> tuple[dict[str, Any], bool]:
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

    if lenient:
        repaired = _lenient(text)
        if repaired is not None:
            return repaired, False
    return _fallback(text), True


def parse_generation_output(text: str, *, lenient: bool = False) -> dict[str, Any]:
    """Parse raw model output into a dict with answer, rationale, confidence, citations.

    Extra keys from the model's JSON are kept, as in the research work. With
    `lenient`, fields are read from invalid JSON before the sentence fallback.
    """
    return _parse(text, lenient=lenient)[0]


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
    def from_text(cls, text: str, *, lenient: bool = False) -> "ParsedGeneration":
        raw, used_fallback = _parse(text, lenient=lenient)
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
