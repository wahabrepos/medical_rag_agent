"""Answer one question: run the agent, assess the evidence, record cost and run.

`stream` yields progress events while the agent works, then the final answer.
"""

import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from medrag_agent.budget import SpendLedger, cost
from medrag_agent.llm import LlmGenerator
from medrag_api.repository import RunRepository
from medrag_api.schemas import (
    EVIDENCE_MESSAGES,
    AskRequest,
    AskResponse,
    Citation,
    EvidenceOut,
    StatementEvidenceOut,
)
from medrag_core.evidence import (
    UNCERTAIN,
    AnswerFormat,
    EvidenceStatus,
    assess_evidence,
    present_answer,
)

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
# Worst case for one question: 3 iterations of a long prompt and a full completion.
WORST_CASE_PROMPT_TOKENS = 3 * 3_000

Event = tuple[str, dict[str, Any]]


class Components(Protocol):
    """What the service needs from medrag_agent.runtime.AgentComponents."""

    nli: Any

    def new_generator(self) -> LlmGenerator: ...

    def new_graph(self, generator: LlmGenerator) -> Any: ...


@dataclass
class AnswerService:
    components: Components
    repository: RunRepository
    ledger: SpendLedger
    allow_uncertain: bool = True
    clock: Callable[[], float] = time.monotonic

    def stream(self, request: AskRequest) -> Iterator[Event]:
        generator = self.components.new_generator()
        model = generator.config.model
        self.ledger.check(cost(model, WORST_CASE_PROMPT_TOKENS, 3 * generator.config.max_tokens))
        graph = self.components.new_graph(generator)
        started = self.clock()
        state: dict[str, Any] = {}
        initial = {
            "question": request.question,
            "binary_answer": request.answer_format is AnswerFormat.YES_NO,
            "multiple_choice": request.answer_format is AnswerFormat.MULTIPLE_CHOICE,
        }
        try:
            for mode, chunk in graph.stream(initial, stream_mode=["updates", "values"]):
                if mode == "values":
                    state = chunk
                    continue
                for node, update in chunk.items():
                    event = _progress_event(node, update or {}, state)
                    if event:
                        yield event
        finally:
            # Charge whatever was spent, even when the run fails part-way.
            spent = self.ledger.add(
                run="api",
                model=model,
                prompt_tokens=generator.prompt_tokens,
                completion_tokens=generator.completion_tokens,
            )

        response = self._response(request, state, model, self.clock() - started, spent)
        payload = response.model_dump(mode="json")
        self.repository.save(response.run_id, payload)
        yield "answer", payload

    def ask(self, request: AskRequest) -> AskResponse:
        answer: dict[str, Any] | None = None
        for name, data in self.stream(request):
            if name == "answer":
                answer = data
        if answer is None:
            raise RuntimeError("the agent produced no answer")
        return AskResponse.model_validate(answer)

    def _response(
        self, request: AskRequest, state: dict[str, Any], model: str, latency: float, spent: float
    ) -> AskResponse:
        final_iteration = state.get("final_iteration")
        passages: Sequence[Any] = (
            state["iteration_passages"][final_iteration - 1] if final_iteration else []
        )
        texts = [p.text for p in passages]
        assessment = assess_evidence(
            state.get("rationale", []),
            texts,
            lambda pairs: self.components.nli.probabilities(pairs),
        )

        def pmid(index: int | None) -> int | None:
            return passages[index].pmid if index is not None else None

        model_answer = state.get("answer", "")
        answer = present_answer(
            model_answer,
            assessment.status,
            request.answer_format,
            allow_uncertain=self.allow_uncertain,
        )
        note = (
            f'No retrieved study supports an answer; the model leans "{model_answer}".'
            if answer == UNCERTAIN and model_answer != UNCERTAIN
            else None
        )
        seen: set[int] = set()
        citations = []
        for p in passages:
            if p.pmid not in seen:
                seen.add(p.pmid)
                citations.append(
                    Citation(
                        pmid=p.pmid,
                        title=p.title,
                        url=PUBMED_URL.format(pmid=p.pmid),
                        passage=p.text,
                    )
                )
        return AskResponse(
            run_id=uuid.uuid4(),
            question=request.question,
            answer_format=request.answer_format,
            answer=answer,
            model_answer=model_answer,
            note=note,
            evidence=EvidenceOut(
                status=assessment.status,
                supported_fraction=assessment.supported_fraction,
                message=EVIDENCE_MESSAGES[assessment.status],
                statements=[
                    StatementEvidenceOut(
                        text=s.statement,
                        support=round(s.support, 4),
                        supported=s.supported,
                        contradicted=s.contradicted,
                        supporting_pmid=pmid(s.supporting_passage),
                        contradicting_pmid=pmid(s.contradicting_passage),
                    )
                    for s in assessment.statements
                ],
            ),
            citations=citations,
            iterations=state.get("iteration", 0),
            stop_reason=state.get("stop_reason", "error"),
            model=model,
            latency_seconds=round(latency, 3),
            cost=round(spent, 6),
        )


def _progress_event(node: str, update: dict[str, Any], state: dict[str, Any]) -> Event | None:
    iteration = state.get("iteration", 0) + 1
    if node == "retrieve" and "context" in update:
        return "retrieved", {"iteration": iteration, "passages": len(update["context"])}
    if node == "generate" and "generation" in update:
        return "generated", {"iteration": iteration, "draft_answer": update["generation"].answer}
    if node == "verify" and "verification" in update:
        return "verified", {
            "iteration": iteration,
            "support": round(update["verification"].support_score, 4),
            "decision": update.get("decision"),
        }
    if node == "refine":
        return "refining", {"next_iteration": update.get("iteration", 0) + 1}
    return None


__all__ = ["AnswerService", "EvidenceStatus"]
