"""HTTP inference service: sentence embeddings and NLI probabilities on CPU.

    uv run --env-file .env uvicorn medrag_inference.app:app --port 8001

Endpoints are plain `def`, so FastAPI runs inference in its thread pool and the
event loop stays free. Models are loaded once, at startup.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Protocol

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from medrag_inference.nli import LABELS

MAX_TEXTS = 256
MAX_PAIRS = 512
MAX_CHARS = 20_000


class Embedder(Protocol):
    @property
    def version(self) -> str: ...

    def embed(self, texts: list[str], *, batch_size: int = ...) -> npt.NDArray[np.float32]: ...


class Nli(Protocol):
    @property
    def version(self) -> str: ...

    def probabilities(
        self, pairs: list[tuple[str, str]], *, batch_size: int = ...
    ) -> npt.NDArray[np.float32]: ...


@dataclass
class Models:
    embedder: Embedder
    nli: Nli


Text = Annotated[str, Field(max_length=MAX_CHARS)]


class EmbedRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS)


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]]


class NliRequest(BaseModel):
    pairs: list[tuple[Text, Text]] = Field(
        min_length=1, max_length=MAX_PAIRS, description="(premise, hypothesis) pairs"
    )


class NliResponse(BaseModel):
    model: str
    labels: list[str]
    probabilities: list[list[float]]


class Health(BaseModel):
    status: str
    embedder: str
    nli: str


def load_models() -> Models:
    from medrag_inference.embedding import BgeEmbedder
    from medrag_inference.nli import DebertaNli
    from medrag_settings import get_settings

    threads = get_settings().inference_threads
    return Models(embedder=BgeEmbedder(threads=threads), nli=DebertaNli(threads=threads))


def create_app(loader: Callable[[], Models] = load_models) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.models = loader()
        yield

    app = FastAPI(title="medrag inference", version="0.1.0", lifespan=lifespan)

    def models(request: Request) -> Models:
        loaded: Models = request.app.state.models
        return loaded

    @app.get("/healthz")
    def healthz(request: Request) -> Health:
        m = models(request)
        return Health(status="ok", embedder=m.embedder.version, nli=m.nli.version)

    @app.post("/embed")
    def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
        m = models(request)
        vectors = m.embedder.embed(body.texts)
        return EmbedResponse(
            model=m.embedder.version, dim=int(vectors.shape[1]), vectors=vectors.tolist()
        )

    @app.post("/nli")
    def nli(body: NliRequest, request: Request) -> NliResponse:
        m = models(request)
        probs = m.nli.probabilities(list(body.pairs))
        return NliResponse(model=m.nli.version, labels=list(LABELS), probabilities=probs.tolist())

    return app


app = create_app()
