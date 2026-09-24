"""Framework-free Self-MedRAG logic: prompts, parsing, fusion, verification, stop rules."""

from medrag_core.loop import (
    Generation,
    IterationRecord,
    LoopResult,
    StopReason,
    run_self_medrag,
)
from medrag_core.parsing import ParsedGeneration, parse_generation_output
from medrag_core.policy import Decision, LoopSettings, RefinementStrategy, decide, refine_query
from medrag_core.prompts import PROMPT_VERSION, ChatMessages, build_messages, build_prompt
from medrag_core.retrieval import RetrievalMode, reciprocal_rank_fusion, select_context, tokenize
from medrag_core.verification import NliScorer, Verification, verify_rationale

__all__ = [
    "PROMPT_VERSION",
    "ChatMessages",
    "Decision",
    "Generation",
    "IterationRecord",
    "LoopResult",
    "LoopSettings",
    "NliScorer",
    "ParsedGeneration",
    "RefinementStrategy",
    "RetrievalMode",
    "StopReason",
    "Verification",
    "build_messages",
    "build_prompt",
    "decide",
    "parse_generation_output",
    "reciprocal_rank_fusion",
    "refine_query",
    "run_self_medrag",
    "select_context",
    "tokenize",
    "verify_rationale",
]
__version__ = "0.1.0"
