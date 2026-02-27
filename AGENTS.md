# AGENTS.md

## Project overview

Zephon is a modular, multimodal-first data loader for ML with deterministic parallel execution and elastic continuation across rank-count changes.

## Setup commands

- Install deps: `uv sync --group dev --group test`
- Format + fix: `make format`
- Lint + typecheck: `make lint`

## Testing instructions

- Unit tests (default): `make test`
- Integration tests: `make integration`
- Single test file: `pytest tests/zephon/core/test_engine.py`
- Named test: `pytest -k "test_name"`

## Python development standards

- Use Python 3.11+ type annotation syntax for new/updated code
- Add type hints for function parameters and return values
- Prefer built-in generics (e.g., `list[str]`) over legacy `typing.List`
- Use `uv run ...` for ad-hoc script/module execution instead of raw `python ...`
- Avoid adding shebangs (e.g., `#!/usr/bin/env python3`) to project Python files

## Key Files

`WorkSource -> Pipeline -> Graph -> Planner -> RuntimeSpec -> Engine`

- `api/pipeline.py`: user API and operator chaining
- `core/planner.py`: graph-to-stage compilation
- `core/engine.py`: runtime execution, queues, shutdown
- `core/op_base.py`: `Op` contract (`setup`, `traits`, `accumulator`, `process_many`)

Critical invariant: `process_many()` must stay stateless across calls; cross-call state belongs in the accumulator.

## Code style and guardrails

- Ruff for lint/format; Pyright for type checking
- Google-style docstrings (ruff `D` rules)
- Keep imports isort-compatible (`zephon` first-party)
- Integration tests are opt-in; run them for runtime/scheduling/I/O changes
- Do not edit generated `zephon/_version.py` (managed by `setuptools-scm`)
