"""The Self-MedRAG loop as a LangGraph state graph.

    START -> prepare -> guard --(ok)--> retrieve -> generate -> verify -> decide
                          ^                                              |
                          |                           continue           |
                          +---------------- refine <---------------------+
                          guard/decide --(stop)--> finalize -> END

One deliberate difference: `ProviderUnavailableError` (quota or rate limit) is not
recorded as an "Error during generation" answer; it propagates so the run stops
and the question is asked again later.

Every node uses the same rules as `medrag_core.loop.run_self_medrag` (thresholds,
early stopping, refinement strategies, best-of-history, timeout and error
handling); the graph is tested against the same research-work fixtures. Being a
graph, each step can be streamed and checkpointed.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medrag_agent.errors import ProviderUnavailableError
from medrag_core.loop import (
    ERROR_ANSWER,
    Generation,
    IterationRecord,
    LoopResult,
    StopReason,
)
from medrag_core.policy import Decision, LoopSettings, choose_final, decide, refine_query
from medrag_core.verification import Verification

Retriever = Callable[..., Sequence[str]]  # (query, *, exclude_pmids=...) -> passages
GeneratorFn = Callable[..., Generation]
Verifier = Callable[[list[str], list[str]], Verification]


class AgentState(TypedDict, total=False):
    question: str  # as given; refinement starts from it unstripped (research-work rule)
    binary_answer: bool  # PubMedQA: the yes/no constraint is added to the prompt
    multiple_choice: bool  # optional: require an option letter (not in the research work)
    exclude_pmids: list[int]  # articles retrieval must skip (leakage-free evaluation)
    current_query: str
    iteration: int  # completed iterations
    started_at: float
    history: list[IterationRecord]
    context: list[str]
    generation: Generation
    verification: Verification
    decision: str
    error: str
    stop_reason: str
    answer: str
    rationale: list[str]
    support_score: float


def build_graph(
    *,
    retrieve: Retriever,
    generate: GeneratorFn,
    verify: Verifier,
    settings: LoopSettings | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    settings = settings or LoopSettings()

    def prepare(state: AgentState) -> AgentState:
        return {
            "current_query": state["question"].strip(),
            "iteration": 0,
            "started_at": clock(),
            "history": [],
        }

    def guard(state: AgentState) -> AgentState:
        if clock() - state["started_at"] > settings.max_time_seconds:
            return {"stop_reason": StopReason.TIMEOUT.value}
        return {}

    def retrieve_node(state: AgentState) -> AgentState:
        try:
            exclude = state.get("exclude_pmids")
            passages = (
                retrieve(state["current_query"], exclude_pmids=exclude)
                if exclude
                else retrieve(state["current_query"])
            )
            return {"context": list(passages)}
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            return {"error": str(exc)}

    def generate_node(state: AgentState) -> AgentState:
        try:
            generation = generate(
                state["current_query"],
                state["context"],
                list(state["history"]),
                binary_answer=state.get("binary_answer", False),
                multiple_choice=state.get("multiple_choice", False),
            )
            return {"generation": generation}
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            return {"error": str(exc)}

    def verify_node(state: AgentState) -> AgentState:
        try:
            generation = state["generation"]
            verification = verify(generation.rationale, state["context"])
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            return {"error": str(exc)}
        record = IterationRecord(
            iteration=state["iteration"] + 1,
            query=state["current_query"],
            context=state["context"],
            answer=generation.answer,
            rationale=generation.rationale,
            confidence=generation.confidence,
            support_score=verification.support_score,
            citations=generation.citations,
        )
        history = [*state["history"], record]
        return {
            "verification": verification,
            "history": history,
            "decision": decide(history, settings).value,
        }

    def refine_node(state: AgentState) -> AgentState:
        iteration = state["iteration"] + 1
        refined = refine_query(
            state["question"], state["verification"].unsupported, iteration, settings
        )
        update: AgentState = {"iteration": iteration}
        if refined is not None:
            update["current_query"] = refined
        return update

    def finalize(state: AgentState) -> AgentState:
        history = state.get("history", [])
        iteration = state.get("iteration", 0)
        answer, rationale, support = "", [], 0.0
        reason = state.get("stop_reason")

        if state.get("error") is not None:
            if iteration > 0:
                last = history[-1]
                answer, rationale, support = last.answer, last.rationale, last.support_score
            else:
                answer, rationale, support = ERROR_ANSWER, [state["error"]], 0.0
            iteration += 1
            reason = StopReason.ERROR.value
        elif state.get("decision") in (Decision.ACCEPT.value, Decision.STALLED.value):
            chosen = (
                history[-1]
                if state["decision"] == Decision.ACCEPT.value
                else choose_final(history, stalled=True, settings=settings)
            )
            answer, rationale, support = chosen.answer, chosen.rationale, chosen.support_score
            iteration += 1
            reason = (
                StopReason.ACCEPTED.value
                if state["decision"] == Decision.ACCEPT.value
                else StopReason.STALLED.value
            )

        if iteration == settings.max_iterations and not answer:
            if history:
                best = choose_final(history, stalled=False, settings=settings)
                answer, rationale, support = best.answer, best.rationale, best.support_score
            else:
                answer, rationale, support = (
                    "Unable to generate answer",
                    ["No valid iterations completed"],
                    0.0,
                )
            reason = StopReason.MAX_ITERATIONS.value

        return {
            "answer": answer,
            "rationale": rationale,
            "support_score": support,
            "iteration": iteration,
            "stop_reason": reason or StopReason.MAX_ITERATIONS.value,
        }

    def after_guard(state: AgentState) -> Literal["retrieve", "finalize"]:
        stop = state.get("stop_reason") or state["iteration"] >= settings.max_iterations
        return "finalize" if stop else "retrieve"

    def on_error(next_node: str) -> Callable[[AgentState], str]:
        def route(state: AgentState) -> str:
            return "finalize" if state.get("error") is not None else next_node

        return route

    def after_verify(state: AgentState) -> Literal["refine", "finalize"]:
        if state.get("error") is not None or state["decision"] != Decision.CONTINUE.value:
            return "finalize"
        return "refine"

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("guard", guard)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("refine", refine_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "guard")
    graph.add_conditional_edges("guard", after_guard, ["retrieve", "finalize"])
    graph.add_conditional_edges("retrieve", on_error("generate"), ["generate", "finalize"])
    graph.add_conditional_edges("generate", on_error("verify"), ["verify", "finalize"])
    graph.add_conditional_edges("verify", after_verify, ["refine", "finalize"])
    graph.add_edge("refine", "guard")
    graph.add_edge("finalize", END)
    return graph.compile()


def to_loop_result(state: dict[str, Any]) -> LoopResult:
    """The graph's final state in the same shape as the core loop's result."""
    return LoopResult(
        answer=state["answer"],
        rationale=state["rationale"],
        iterations=state["iteration"],
        support_score=state["support_score"],
        stop_reason=StopReason(state["stop_reason"]),
        history=list(state.get("history", [])),
    )
