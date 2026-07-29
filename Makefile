# Project makefile (uv + pytest)

UV ?= uv
PY ?= $(UV) run
PYTEST ?= pytest
EXTRA_ARGS ?=

.PHONY: help setup test integration packaging lint format docs typecheck-public verifytypes

help:
	@echo "Targets:"
	@echo "  setup           - Create dev/test environment via uv"
	@echo "  test            - Run pytest ($(EXTRA_ARGS) optional)"
	@echo "  integration     - Run integration tests"
	@echo "  packaging       - Build the wheel/sdist and smoke-test the artifacts"
	@echo "  lint            - Check formatting, linting, and types"
	@echo "  format          - Format code and fix lint issues"
	@echo "  docs            - Build HTML documentation"
	@echo "  typecheck-public - Type-check the public-API consumer fixture"
	@echo "  verifytypes     - Public-API type-completeness / leak gate"

setup:
	$(UV) sync --group dev --group test

test:
	$(PY) $(PYTEST) $(EXTRA_ARGS)

integration:
	$(PY) $(PYTEST) --run-integration tests/integration $(EXTRA_ARGS)

packaging:
	$(PY) $(PYTEST) --run-packaging tests/packaging $(EXTRA_ARGS)

lint:
	./linting/lint.sh --check

format:
	./linting/lint.sh

docs:
	cd docs && $(MAKE) html

# The default pyright run only covers zephon/**; this checks the public-API
# consumer fixture separately.
typecheck-public:
	$(PY) pyright tests/typing

# Public-API type-completeness / leak gate. Run against the installed wheel
# (see tests/packaging); editable installs report "no py.typed found".
verifytypes:
	$(PY) pyright --verifytypes zephon --ignoreexternal
