# Project makefile (uv + pytest)

UV ?= uv
PY ?= $(UV) run
PYTEST ?= pytest
EXTRA_ARGS ?=

.PHONY: help setup test integration lint format docs

help:
	@echo "Targets:"
	@echo "  setup        - Create dev/test environment via uv"
	@echo "  test         - Run pytest ($(EXTRA_ARGS) optional)"
	@echo "  integration  - Run integration tests"
	@echo "  lint         - Check formatting, linting, and types"
	@echo "  format       - Format code and fix lint issues"
	@echo "  docs         - Build HTML documentation"

setup:
	$(UV) sync --group dev --group test

test:
	$(PY) $(PYTEST) $(EXTRA_ARGS)

integration:
	$(PY) $(PYTEST) --run-integration tests/integration $(EXTRA_ARGS)

lint:
	./linting/lint.sh --check

format:
	./linting/lint.sh

docs:
	cd docs && $(MAKE) html
