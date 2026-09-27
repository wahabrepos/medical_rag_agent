.PHONY: install lint fmt typecheck test check hooks web-install web-dev web-check

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
