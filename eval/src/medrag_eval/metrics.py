"""Answer extraction and benchmark metrics, ported from the research-work evaluator.

Accuracy, macro F1 and latency percentiles follow scikit-learn and numpy
semantics (macro F1 over every label seen in either list, zero division -> 0;
linear-interpolated percentiles) without depending on either library.
"""

import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any


class Dataset(StrEnum):
    MEDQA = "medqa"
    PUBMEDQA = "pubmedqa"


def normalize_text(text: str) -> str:
    return text.strip().lower()


def extract_option_letter(text: str) -> str:
    """First A-E option letter in a model answer, uppercased, or "" if none.

    Handles "A", "A.", "A)", "(A)", "The answer is A", "A. Penicillin", ...
    """
    m = re.match(r"^\s*([A-Ea-e])[.\):\s]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-Ea-e])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"[(\[]([A-Ea-e])[)\]]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Ea-e])\b", text)
    if m:
        return m.group(1).upper()
    return ""


def extract_binary_answer(text: str) -> str:
    """ "yes" or "no" from a PubMedQA answer; otherwise the lowercased text itself."""
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    m = re.match(r"^(yes|no)[^a-z]", t)
    if m:
        return m.group(1)
    for word in ("yes", "no"):
        if re.search(rf"\b{word}\b", t):
            return word
    return t


def normalize_prediction(answer: str, dataset: Dataset) -> str:
    if dataset is Dataset.MEDQA:
        return (extract_option_letter(answer) or normalize_text(answer)).lower()
    return extract_binary_answer(answer)


def normalize_ground_truth(answer: str, dataset: Dataset) -> str:
    gt = normalize_text(answer)
    return gt.lower() if dataset is Dataset.MEDQA else gt


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    total = 0.0
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        denom = 2 * tp + fp + fn
        total += 2 * tp / denom if denom else 0.0
    return total / len(labels)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def evaluate(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[str],
    dataset: Dataset,
) -> dict[str, float]:
    """Metrics for one dataset, as reported by the research work.

    Each prediction needs `final_answer`; `iterations` (default 1), `support_score`,
    `confidence` (default 0) and `latency` are used when present.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    y_pred = [normalize_prediction(p.get("final_answer", ""), dataset) for p in predictions]
    y_true = [normalize_ground_truth(g, dataset) for g in ground_truth]
    iterations = [p.get("iterations", 1) for p in predictions]
    latencies = [p["latency"] for p in predictions if p.get("latency") is not None]

    results = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_score": macro_f1(y_true, y_pred),
        "avg_iterations": mean(iterations),
        "more_than_one_iter_pct": sum(i > 1 for i in iterations) / len(iterations) * 100,
        "avg_support_score": mean([p.get("support_score", 0.0) for p in predictions]),
        "avg_confidence": mean([p.get("confidence", 0.0) for p in predictions]),
    }
    if latencies:
        results["latency_p50"] = percentile(latencies, 50)
        results["latency_p95"] = percentile(latencies, 95)
    return results
