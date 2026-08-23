# Makefile - Convenience commands for the project
.PHONY: vulture vulture-ci install test

# Run vulture locally (shows all findings)
vulture:
	./venv/bin/vulture src/ app/ scripts/

# Run vulture for CI (same as vulture, vulture exits 0 by default)
vulture-ci:
	./venv/bin/vulture src/ app/ scripts/

# Install dev dependencies (poetry)
install:
	poetry install --with dev

# Run all linting (ruff + pyright + vulture)
lint:
	./venv/bin/ruff check src/ app/ scripts/
	./venv/bin/pyright
	$(MAKE) vulture-ci

# Quick dev setup (venv + deps + data)
setup:
	./scripts/run.sh --skip-setup