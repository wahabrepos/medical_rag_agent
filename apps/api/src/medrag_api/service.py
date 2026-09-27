"""Answer one question: run the agent, assess the evidence, record cost and run.

`stream` yields progress events while the agent works, then the final answer.
"""

import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from medrag_agent.assessment import assess_final
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
    AnswerFormat,
    AnswerPolicy,
    EvidenceStatus,
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
    policy: AnswerPolicy = AnswerPolicy.EVIDENCE_GATED
    clock: Callable[[], float] = time.monotonic

    def stream(self, request: AskRequest) -> Iterator[Event]:
        generator = self.components.new_generator()
        model = generator.config.model
        self.ledger.check(cost(model, WORST_CASE_PROMPT_TOKENS, 3 * generator.config.max_tokens))
        graph = self.components.new_graph(generator)
        started = self.clock()
        # A draft answer is unverified: with the evidence gate it is only streamed
        # when the caller asked for unverified answers.
        show_drafts = self.policy is not AnswerPolicy.EVIDENCE_GATED or request.include_unverified
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
                    event = _progress_event(node, update or {}, state, show_drafts=show_drafts)
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
        final = assess_final(
            state, self.components.nli, quoted=getattr(self.components, "evidence_quotes", False)
        )
        assessment, passages, claim = final.assessment, final.passages, final.has_claim

        def pmid(index: int | None) -> int | None:
            return (passages[index].pmid or None) if index is not None else None

        model_answer = state.get("answer", "")
        answer = present_answer(model_answer, assessment, request.answer_format, policy=self.policy)
        withheld = answer != model_answer
        note = None
        if withheld:
            note = (
                "No answer is shown because the retrieved studies do not support every part "
                "of it; the studies found are listed below."
            )
            if request.include_unverified:
                note += f' Unverified model answer (not backed by these studies): "{model_answer}".'
        # One citation per distinct passage, so a quote always points at the text it
        # came from (two passages of one article are two citations).
        citations: list[Citation] = []
        citation_of: dict[int, int] = {}  # passage index -> citation index
        seen: dict[tuple[str, object], int] = {}
        for i, p in enumerate(passages):
            source = getattr(p, "source", "pubmed")
            key = (source, getattr(p, "chunk_id", p.pmid))
            if key not in seen:
                seen[key] = len(citations)
                citations.append(
                    Citation(
                        source=source,
                        pmid=p.pmid if source == "pubmed" else None,
                        title=p.title,
                        url=PUBMED_URL.format(pmid=p.pmid) if source == "pubmed" else None,
                        passage=p.text,
                    )
                )
            citation_of[i] = seen[key]
        return AskResponse(
            run_id=uuid.uuid4(),
            question=request.question,
            answer_format=request.answer_format,
            answer=answer,
            model_answer=None if withheld and not request.include_unverified else model_answer,
            note=note,
            evidence=EvidenceOut(
                status=assessment.status,
                supported_fraction=assessment.supported_fraction,
                message=EVIDENCE_MESSAGES[assessment.status],
                statements=[
                    StatementEvidenceOut(
                        kind="claim" if claim and i == 0 else "rationale",
                        text=s.statement,
                        support=round(s.support, 4),
                        supported=s.supported,
                        contradicted=s.contradicted,
                        supporting_pmid=pmid(s.supporting_passage),
                        supporting_citation=(
                            citation_of[s.supporting_passage]
                            if s.supporting_passage is not None
                            else None
                        ),
                        contradicting_pmid=pmid(s.contradicting_passage),
                        quote=s.quote,
                        from_question=s.from_question,
                    )
                    for i, s in enumerate(assessment.statements)
                ],
            ),
            citations=citations,
            iterations=state.get("iteration", 0),
            stop_reason=state.get("stop_reason", "error"),
            model=model,
            latency_seconds=round(latency, 3),
            cost=round(spent, 6),
        )


def _progress_event(
    node: str, update: dict[str, Any], state: dict[str, Any], *, show_drafts: bool
) -> Event | None:
    iteration = state.get("iteration", 0) + 1
    if node == "retrieve" and "context" in update:
        return "retrieved", {"iteration": iteration, "passages": len(update["context"])}
    if node == "generate" and "generation" in update:
        data: dict[str, Any] = {"iteration": iteration}
        if show_drafts:
            data["draft_answer"] = update["generation"].answer
        return "generated", data
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
