# Project makefile (uv + ruff + pytest + sphinx)

UV ?= uv
PY ?= $(UV) run
PYTEST ?= pytest
RUFF ?= ruff
PYRIGHT ?= pyright
EXTRA_ARGS ?=

.PHONY: help setup test lint format typecheck docs-html docs-clean docs-serve

help:
	@echo "Targets:"
	@echo "  setup        - Create dev/test environment via uv"
	@echo "  test         - Run pytest ($(EXTRA_ARGS) optional)"
	@echo "  lint         - Ruff format --check and lint"
	@echo "  format       - Ruff format and fix lint"
    @echo "  typecheck    - Pyright type checking"
	@echo "  docs-html    - Build Sphinx HTML docs"
	@echo "  docs-clean   - Remove docs/_build"
	@echo "  docs-serve   - Serve built docs locally"

setup:
	$(UV) sync --group dev --group test

test:
	$(PY) $(PYTEST) $(EXTRA_ARGS)

lint:
	$(PY) $(RUFF) format --check . --exclude docs/_build
	$(PY) $(RUFF) check . --exclude docs/_build

format:
	$(PY) $(RUFF) format . --exclude docs/_build
	$(PY) $(RUFF) check . --fix --exclude docs/_build

typecheck:
    $(PY) $(PYRIGHT)

docs-html:
	$(PY) sphinx-build -b html docs/source docs/_build/html

docs-clean:
	rm -rf docs/_build

docs-serve:
	cd docs/_build/html && $(PY) python -m http.server 8000