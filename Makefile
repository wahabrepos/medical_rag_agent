.PHONY: install lint fmt typecheck test check hooks

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
