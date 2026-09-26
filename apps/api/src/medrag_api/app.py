"""Public HTTP API for the Self-MedRAG agent.

    uv run --env-file .env uvicorn medrag_api.app:app --port 8000

    GET  /healthz                     liveness
    GET  /readyz                      dependencies ready (database, NLI, LLM key, budget)
    POST /v1/ask                      answer as JSON
    POST /v1/ask/stream               progress events, then the answer (Server-Sent Events)
    GET  /v1/runs/{run_id}            a stored answer
    POST /v1/runs/{run_id}/feedback   rating (+1 / -1) and optional comment

Requests need `Authorization: Bearer <key>` (API_KEYS); only APP_ENV=local may run
without keys. Each key is rate limited, and answering stops with 503 before total LLM
spend could pass 90% of LLM_BUDGET.
"""

import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from medrag_agent.budget import BudgetExceededError
from medrag_agent.errors import ProviderUnavailableError
from medrag_api.schemas import AskRequest, AskResponse, FeedbackRequest
from medrag_api.service import AnswerService

logger = logging.getLogger(__name__)


class KeyRateLimiter:
    """Sliding one-minute window of requests per API key."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= 60.0:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


class ApiContext:
    def __init__(
        self,
        service: AnswerService,
        *,
        api_keys: set[str],
        open_access: bool,
        requests_per_minute: int,
        readiness: Callable[[], dict[str, str]] = dict,
    ) -> None:
        self.service = service
        self.api_keys = api_keys
        self.open_access = open_access
        self.limiter = KeyRateLimiter(requests_per_minute)
        self.readiness = readiness


def load_context() -> ApiContext:
    from pathlib import Path

    from medrag_agent.budget import SpendLedger
    from medrag_agent.runtime import PRODUCT_LOOP, build_components
    from medrag_api.repository import SqlRunRepository
    from medrag_db.session import make_engine, make_session_factory
    from medrag_settings import AppEnv, get_settings

    settings = get_settings()
    components = build_components(
        settings,
        support_label="entailment",
        loop=PRODUCT_LOOP,
        lenient_json=True,
        nli_url=settings.nli_url,
    )
    sessions = make_session_factory(make_engine(settings.database_url.get_secret_value()))
    service = AnswerService(
        components=components,
        repository=SqlRunRepository(sessions),
        ledger=SpendLedger(Path(settings.llm_spend_ledger), cap=settings.llm_budget),
        allow_uncertain=settings.allow_uncertain,
    )
    keys = settings.api_keys.get_secret_value() if settings.api_keys else ""

    def readiness() -> dict[str, str]:
        from sqlalchemy import text

        checks = {}
        with sessions() as session:
            session.execute(text("select 1"))
        checks["database"] = "ok"
        checks["nli"] = getattr(components.nli, "version", "in-process")
        if components.new_generator().api_key is None:
            raise RuntimeError("no API key for the LLM provider")
        checks["llm"] = settings.llm_primary_model
        remaining = service.ledger.limit - service.ledger.total
        if remaining <= 0:
            raise RuntimeError("LLM budget used up")
        checks["llm_budget_remaining"] = f"{remaining:.2f}"
        return checks

    return ApiContext(
        service,
        api_keys={k.strip() for k in keys.split(",") if k.strip()},
        open_access=settings.app_env is AppEnv.LOCAL and not keys,
        requests_per_minute=settings.api_requests_per_minute,
        readiness=readiness,
    )


def context(request: Request) -> ApiContext:
    ctx: ApiContext = request.app.state.ctx
    return ctx


Ctx = Annotated[ApiContext, Depends(context)]


def caller(ctx: Ctx, authorization: Annotated[str | None, Header()] = None) -> str:
    """The API key making the request, after authentication and rate limiting."""
    key = (authorization or "").removeprefix("Bearer ").strip()
    if ctx.open_access and not key:
        key = "local"
    elif key not in ctx.api_keys:
        raise HTTPException(401, "missing or invalid API key")
    if not ctx.limiter.allow(key):
        raise HTTPException(429, "rate limit exceeded; try again in a minute")
    return key


Caller = Annotated[str, Depends(caller)]


def create_app(loader: Callable[[], ApiContext] = load_context) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = loader()
        yield

    app = FastAPI(title="medical_rag_agent", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(ctx: Ctx) -> dict[str, Any]:
        try:
            return {"status": "ready", "checks": ctx.readiness()}
        except Exception as exc:
            raise HTTPException(503, f"not ready: {exc}") from exc

    @app.post("/v1/ask")
    def ask(body: AskRequest, ctx: Ctx, _: Caller) -> AskResponse:
        try:
            return ctx.service.ask(body)
        except (BudgetExceededError, ProviderUnavailableError) as exc:
            raise HTTPException(503, f"temporarily unavailable: {exc}") from exc

    @app.post("/v1/ask/stream")
    def ask_stream(body: AskRequest, ctx: Ctx, _: Caller) -> StreamingResponse:
        def events() -> Iterator[str]:
            try:
                for name, data in ctx.service.stream(body):
                    yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
            except (BudgetExceededError, ProviderUnavailableError) as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            except Exception:
                logger.exception("answer stream failed")
                yield f"event: error\ndata: {json.dumps({'detail': 'internal error'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: uuid.UUID, ctx: Ctx, _: Caller) -> dict[str, Any]:
        run = ctx.service.repository.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    @app.post("/v1/runs/{run_id}/feedback", status_code=201)
    def feedback(run_id: uuid.UUID, body: FeedbackRequest, ctx: Ctx, _: Caller) -> dict[str, str]:
        if body.rating not in (-1, 1):
            raise HTTPException(422, "rating must be 1 or -1")
        if not ctx.service.repository.add_feedback(run_id, body.rating, body.comment):
            raise HTTPException(404, "run not found")
        return {"status": "recorded"}

    return app


app = create_app()
