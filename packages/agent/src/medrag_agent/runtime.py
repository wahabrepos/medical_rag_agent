"""Wire the Self-MedRAG agent to real components.

`AgentComponents` holds what is shared and thread-safe (retriever, NLI, rate
limiter); each question gets its own generator, so its token cost is counted
exactly even when several questions run at once.

`build_parity_agent` keeps the research-work pipeline for evaluations: hybrid
retrieval over the 10k-record parity corpus, DeBERTa NLI reading the research
work's support column ("neutral") unless told otherwise, the research-work prompts
and loop settings, and the configured generator (see ADR-0008).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from medrag_agent.errors import ProviderUnavailableError
from medrag_agent.graph import AgentState, build_graph
from medrag_agent.llm import LlmGenerator, RateLimiter, config_for_model
from medrag_core.policy import FinalAnswerRule, LoopSettings
from medrag_core.verification import (
    Verification,
    verify_rationale,
    verify_with_contradictions,
)
from medrag_db.session import make_engine, make_session_factory
from medrag_inference import RESEARCH_WORK_SUPPORT_LABEL, BgeEmbedder
from medrag_inference.nli import LabelScorer, load_verifier
from medrag_search import BM25Index, HybridRetriever, LiveAugmentedRetriever, PubMedClient
from medrag_settings import Settings

PARITY_PROFILE = "parity"

# The configuration that did best in evaluation (v2d; see the README).
PRODUCT_LOOP = LoopSettings(
    final_answer_rule=FinalAnswerRule.BEST_SUPPORTED, prefer_committed_answers=True
)


class PassageRetriever(Protocol):
    def passages(self, query: str, *, exclude_pmids: Sequence[int] = ...) -> list[Any]: ...

    def __call__(self, query: str, *, exclude_pmids: Sequence[int] = ...) -> list[str]: ...


class NliModel(Protocol):
    # Support probability that counts as supported (0.7 for DeBERTa, 0.5 for MiniCheck).
    support_threshold: float

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = ...
    ) -> npt.NDArray[np.float32]: ...

    def scorer(self, label: str) -> LabelScorer: ...


def api_key_for(model: str, settings: Settings) -> str | None:
    provider = model.split("/", 1)[0]
    secret = {"groq": settings.groq_api_key, "mistral": settings.mistral_api_key}.get(provider)
    return secret.get_secret_value() if secret else None


@dataclass
class AgentComponents:
    settings: Settings
    retriever: PassageRetriever
    nli: NliModel
    support_label: str
    loop: LoopSettings | None
    lenient_json: bool
    normalize_statements: bool = False
    # v3b: answer claim + contradiction veto (entailment support only).
    answer_check: bool = False
    # 7b: claims with verbatim quotes; only quoted, verified claims count as supported.
    evidence_quotes: bool = False
    limiter: RateLimiter = field(init=False)

    def __post_init__(self) -> None:
        # MiniCheck has no neutral class: its "neutral" column is P(unsupported).
        if "MiniCheck" in getattr(self.nli, "version", "") and self.support_label != "entailment":
            raise ValueError('the MiniCheck verifier needs support_label="entailment"')
        self.limiter = RateLimiter(
            self.settings.llm_requests_per_minute, self.settings.llm_tokens_per_minute
        )

    def new_generator(self) -> LlmGenerator:
        model = self.settings.llm_primary_model
        return LlmGenerator(
            config_for_model(
                model,
                requests_per_minute=self.settings.llm_requests_per_minute,
                tokens_per_minute=self.settings.llm_tokens_per_minute,
                lenient_json=self.lenient_json,
                answer_claim=self.answer_check,
                evidence_quotes=self.evidence_quotes,
            ),
            api_key=api_key_for(model, self.settings),
            limiter=self.limiter,
        )

    def verify(self, rationale: list[str], context: list[str]) -> Verification:
        from medrag_inference.client import InferenceUnavailableError

        scorer = self.nli.scorer(self.support_label)
        try:
            if self.answer_check:
                return verify_with_contradictions(
                    rationale,
                    context,
                    lambda pairs: self.nli.probabilities(pairs).tolist(),
                    threshold=self.nli.support_threshold,
                    normalize=self.normalize_statements,
                )
            return verify_rationale(
                rationale,
                context,
                scorer,
                threshold=self.nli.support_threshold,
                normalize=self.normalize_statements,
            )
        except InferenceUnavailableError as exc:
            # A lost tunnel or remote service stops the run; the question is asked again.
            raise ProviderUnavailableError(str(exc)) from exc

    def new_graph(self, generator: LlmGenerator) -> Any:
        return build_graph(
            retrieve=self.retriever.passages,
            generate=generator,
            verify=self.verify,
            settings=self.loop,
        )


def build_components(
    settings: Settings,
    *,
    index_dir: Path = Path("data/indexes"),
    support_label: str = RESEARCH_WORK_SUPPORT_LABEL,
    loop: LoopSettings | None = None,
    lenient_json: bool = False,
    nli_url: str | None = None,
    normalize_statements: bool = False,
    answer_check: bool = False,
    live_pubmed: bool = False,
    evidence_quotes: bool = False,
) -> AgentComponents:
    sessions = make_session_factory(make_engine(settings.database_url.get_secret_value()))
    embedder = BgeEmbedder(threads=settings.inference_threads)
    local = HybridRetriever(
        sessions,
        BM25Index.load(index_dir / f"bm25_{PARITY_PROFILE}.npz"),
        embedder.embed_query,
        profile=PARITY_PROFILE,
    )
    retriever: PassageRetriever = local
    if live_pubmed:
        key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
        retriever = LiveAugmentedRetriever(
            local, PubMedClient(email=settings.ncbi_email, api_key=key)
        )
    nli: NliModel
    if nli_url:
        from medrag_inference.client import RemoteNli

        nli = RemoteNli(nli_url)
    else:
        nli = load_verifier(settings)
    return AgentComponents(
        settings=settings,
        retriever=retriever,
        nli=nli,
        support_label=support_label,
        loop=loop,
        lenient_json=lenient_json,
        normalize_statements=normalize_statements,
        answer_check=answer_check,
        evidence_quotes=evidence_quotes,
    )


@dataclass
class ParityAgent:
    """One generator for a whole evaluation run (the runner is single-threaded)."""

    components: AgentComponents
    generator: LlmGenerator
    graph: Any

    @property
    def nli(self) -> NliModel:
        return self.components.nli

    @property
    def retriever(self) -> PassageRetriever:
        return self.components.retriever

    def run(
        self,
        question: str,
        *,
        binary_answer: bool,
        multiple_choice: bool = False,
        exclude_pmids: list[int] | None = None,
    ) -> AgentState:
        request: AgentState = {
            "question": question,
            "binary_answer": binary_answer,
            "multiple_choice": multiple_choice,
        }
        if exclude_pmids:
            request["exclude_pmids"] = exclude_pmids
        state: AgentState = self.graph.invoke(request)
        return state


def build_parity_agent(settings: Settings, **kwargs: Any) -> ParityAgent:
    components = build_components(settings, **kwargs)
    generator = components.new_generator()
    return ParityAgent(components, generator, components.new_graph(generator))
