"""Wire the Self-MedRAG agent to real components.

The parity agent reproduces the research-work pipeline: hybrid retrieval over the
10k-record parity corpus, DeBERTa NLI reading the research work's support column
("neutral"), the research-work prompts and loop settings, and the configured
generator (Groq gpt-oss-120b by default; see ADR-0008).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrag_agent.errors import ProviderUnavailableError
from medrag_agent.graph import AgentState, build_graph
from medrag_agent.llm import LlmGenerator, config_for_model
from medrag_core.policy import LoopSettings
from medrag_core.verification import VERIFICATION_THRESHOLD, Verification, verify_rationale
from medrag_db.session import make_engine, make_session_factory
from medrag_inference import RESEARCH_WORK_SUPPORT_LABEL, BgeEmbedder, DebertaNli
from medrag_search import BM25Index, HybridRetriever
from medrag_settings import Settings

PARITY_PROFILE = "parity"


@dataclass
class ParityAgent:
    graph: Any
    generator: LlmGenerator
    retriever: HybridRetriever
    nli: object

    def run(
        self, question: str, *, binary_answer: bool, multiple_choice: bool = False
    ) -> AgentState:
        state: AgentState = self.graph.invoke(
            {
                "question": question,
                "binary_answer": binary_answer,
                "multiple_choice": multiple_choice,
            }
        )
        return state


def api_key_for(model: str, settings: Settings) -> str | None:
    provider = model.split("/", 1)[0]
    secret = {"groq": settings.groq_api_key, "mistral": settings.mistral_api_key}.get(provider)
    return secret.get_secret_value() if secret else None


def build_parity_agent(
    settings: Settings,
    *,
    index_dir: Path = Path("data/indexes"),
    support_label: str = RESEARCH_WORK_SUPPORT_LABEL,
    loop: LoopSettings | None = None,
    lenient_json: bool = False,
    nli_url: str | None = None,
) -> ParityAgent:
    sessions = make_session_factory(make_engine(settings.database_url.get_secret_value()))
    embedder = BgeEmbedder(threads=settings.inference_threads)
    retriever = HybridRetriever(
        sessions,
        BM25Index.load(index_dir / f"bm25_{PARITY_PROFILE}.npz"),
        embedder.embed_query,
        profile=PARITY_PROFILE,
    )
    if nli_url:
        from medrag_inference.client import RemoteNli

        nli: RemoteNli | DebertaNli = RemoteNli(nli_url)
    else:
        nli = DebertaNli(threads=settings.inference_threads)
    scorer = nli.scorer(support_label)

    def verify(rationale: list[str], context: list[str]) -> Verification:
        from medrag_inference.client import InferenceUnavailableError

        try:
            return verify_rationale(rationale, context, scorer, threshold=VERIFICATION_THRESHOLD)
        except InferenceUnavailableError as exc:
            # A lost tunnel or remote service stops the run; the question is asked again.
            raise ProviderUnavailableError(str(exc)) from exc

    model = settings.llm_primary_model
    generator = LlmGenerator(
        config_for_model(
            model,
            requests_per_minute=settings.llm_requests_per_minute,
            tokens_per_minute=settings.llm_tokens_per_minute,
            lenient_json=lenient_json,
        ),
        api_key=api_key_for(model, settings),
    )
    graph = build_graph(retrieve=retriever, generate=generator, verify=verify, settings=loop)
    return ParityAgent(graph=graph, generator=generator, retriever=retriever, nli=nli)
