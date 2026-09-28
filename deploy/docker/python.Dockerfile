# syntax=docker/dockerfile:1
# One image recipe for the Python services; PACKAGE picks what gets installed:
#   medrag-api        API (+ Alembic migrations in db/)
#   medrag-inference  embeddings + claim verifier (CPU)
#   medrag-ingest     corpus ingestion job
# Built natively for the machine's platform (linux/arm64 on the Jetson, linux/amd64 on a
# laptop); models are downloaded at run time into the Hugging Face cache volume.
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM python:3.12-slim-bookworm
ARG PACKAGE
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1
WORKDIR /app

# Dependencies first (cached until pyproject.toml or uv.lock change).
COPY pyproject.toml uv.lock ./
COPY packages/settings/pyproject.toml packages/settings/
COPY packages/core/pyproject.toml packages/core/
COPY packages/db/pyproject.toml packages/db/
COPY packages/search/pyproject.toml packages/search/
COPY packages/agent/pyproject.toml packages/agent/
COPY apps/api/pyproject.toml apps/api/
COPY services/inference/pyproject.toml services/inference/
COPY workers/ingest/pyproject.toml workers/ingest/
COPY eval/pyproject.toml eval/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package "$PACKAGE"

# Then the code.
COPY packages packages
COPY apps/api apps/api
COPY services/inference services/inference
COPY workers/ingest workers/ingest
COPY eval eval
COPY db db
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package "$PACKAGE"

# Unprivileged user; uid 1000 matches the usual first host user, so bind-mounted
# caches stay writable. Mount points exist so named volumes inherit the ownership.
RUN useradd --uid 1000 --create-home app \
    && mkdir -p /app/data/indexes /app/host-data /cache/huggingface \
    && chown -R app:app /app/data /app/host-data /cache
USER app
