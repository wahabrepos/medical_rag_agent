"""Benchmark loaders, metrics and the evaluation gate."""

from medrag_eval.datasets import EvalItem, format_medqa_question, load_golden, load_manifest
from medrag_eval.metrics import (
    Dataset,
    evaluate,
    extract_binary_answer,
    extract_option_letter,
)

__all__ = [
    "Dataset",
    "EvalItem",
    "evaluate",
    "extract_binary_answer",
    "extract_option_letter",
    "format_medqa_question",
    "load_golden",
    "load_manifest",
]
__version__ = "0.1.0"
