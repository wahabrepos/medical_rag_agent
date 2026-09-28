.PHONY: install lint fmt typecheck test check hooks web-install web-dev web-check up down clean ingest-sample eval-smoke stack-stats stack-minicheck

install:            ## Install all workspace packages and dev tools
	uv sync --all-packages

hooks:              ## Install git pre-commit hooks
	uv run pre-commit install

lint:               ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

fmt:                ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck:          ## Static type checks
	uv run mypy

test:               ## Run unit tests
	uv run pytest

check: lint typecheck test   ## Everything CI runs (except gitleaks)

web-install:        ## Install the web UI's dependencies (Node.js + corepack)
	cd apps/web && corepack enable && pnpm install

web-dev:            ## Web UI in demo mode (recorded answers) on http://localhost:3000
	cd apps/web && NEXT_PUBLIC_API_URL=mock pnpm dev

web-check:          ## Web UI lint, types, tests and static build (what CI runs)
	cd apps/web && pnpm lint && pnpm typecheck && pnpm test && NEXT_PUBLIC_API_URL=mock pnpm build

# --- Full stack (deploy/docker-compose.yml) -------------------------------------------
COMPOSE = docker compose -f deploy/docker-compose.yml
MEDRAG_PORT ?= 8090
# Reuse the host's Hugging Face cache when there is one (models already downloaded).
HF_CACHE_DIR ?= $(if $(wildcard $(HOME)/.cache/huggingface),$(HOME)/.cache/huggingface,)
export MEDRAG_PORT HF_CACHE_DIR

up:                 ## Build and start the whole stack on http://127.0.0.1:$(MEDRAG_PORT)
	$(COMPOSE) up -d --build

ingest-sample:      ## Load a corpus sample (SAMPLE_RECORDS, default 2000), then restart the API
	$(COMPOSE) run --rm worker
	$(COMPOSE) restart api

eval-smoke:         ## End-to-end check through the web entry point (about EUR 0.001)
	uv run python eval/scripts/smoke_stack.py --url http://127.0.0.1:$(MEDRAG_PORT)

stack-minicheck:    ## Point the API at MiniCheck tunnelled to 172.17.0.1:18001 (deploy/README.md)
	MEDRAG_NLI_URL=http://host.docker.internal:18001 $(COMPOSE) up -d --no-deps api

stack-stats:        ## Memory and CPU per container
	docker stats --no-stream

down:               ## Stop the stack (data volumes are kept)
	$(COMPOSE) down

clean:              ## Stop the stack and delete its volumes (database, index)
	$(COMPOSE) down --volumes
