# syntax=docker/dockerfile:1
# The web UI as static files, served by Caddy together with a reverse proxy to the API
# (same origin, so the browser needs no CORS and the UI calls /v1/... directly).
FROM node:24-bookworm-slim AS build
ENV NEXT_TELEMETRY_DISABLED=1 COREPACK_ENABLE_DOWNLOAD_PROMPT=0
WORKDIR /web
RUN corepack enable
COPY apps/web/package.json apps/web/pnpm-lock.yaml apps/web/pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store pnpm install --frozen-lockfile
COPY apps/web ./
ARG NEXT_PUBLIC_API_URL=/
RUN NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" pnpm build

FROM caddy:2-alpine
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /web/out /srv
