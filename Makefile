.PHONY: install test lint format check doctor

export LITELLM_LOCAL_MODEL_COST_MAP := True
export OPENHANDS_SUPPRESS_BANNER := 1

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

doctor:
	uv run python scripts/doctor.py

check: lint test
