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
- Single test file: `pytest tests/zephon/_internal/test_engine.py`
- Named test: `pytest -k "test_name"`

## Python development standards

- Target Python 3.10+ (the real floor — `requires-python = ">=3.10"`, and CI runs 3.10-3.14 plus 3.14t); use modern built-in-generic annotation syntax but no 3.11-only stdlib
- Add type hints for function parameters and return values
- Prefer built-in generics (e.g., `list[str]`) over legacy `typing.List`
- Use `uv run ...` for ad-hoc script/module execution instead of raw `python ...`
- Avoid adding shebangs (e.g., `#!/usr/bin/env python3`) to project Python files

## Public API vs. internals

We explicitly separate the publicly available API and internal functions that consumers of Zephon should not touch. As a consumer, **never import from or modify anything under `zephon/_internal/`**, and never add operators by touching the graph/planner/engine. Rather, extend via `Pipeline.add_op()` / `Pipeline.map_transform()` or a custom `BaseOp`.

`zephon/_internal/` holds the graph/planner/engine, runners, checkpoint schemas,
IO machinery, built-in op implementations, and utils. It is not API; its layout
and signatures change without notice. The boundary is CI-enforced by
`tests/zephon/test_public_surface.py` (the executable API manifest) and
`tests/zephon/test_internal_boundary.py`. When editing a public module's
`__all__`, update the snapshot in `test_public_surface.py` consciously.

## Key Files

`WorkSource -> Pipeline -> Graph -> Planner -> RuntimeSpec -> Engine`

- `pipeline.py`: user API and operator chaining
- `_internal/planner.py`: graph-to-stage compilation
- `_internal/engine.py`: runtime execution, queues, shutdown
- `ops/base.py`: `BaseOp` authoring contract (`setup`, `traits`, `accumulator`, `process_many`)

Critical invariant: `process_many()` must stay stateless across calls; cross-call state belongs in the accumulator.

## Code style and guardrails

- Ruff for lint/format; Pyright for type checking
- Google-style docstrings (ruff `D` rules)
- Keep imports isort-compatible (`zephon` first-party)
- Integration tests are opt-in; run them for runtime/scheduling/I/O changes
- Do not edit generated `zephon/_version.py` (managed by `setuptools-scm`)
