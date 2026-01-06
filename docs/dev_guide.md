# Zephon Developer Guide

## What is Zephon?

Zephon is a high-performance, modular, multimodal-first data loader for PyTorch and ML applications. It provides:

- Fluent builder-pattern API for composing data pipelines
- Deterministic checkpoint/restart for fault-tolerant training
- Cloud storage support (S3, GCS, local filesystem)
- Elastic training (change rank count while preserving order)
- Multiple data formats (JSONL, LitData, MosaicML Streaming/MDS)

## Prerequisites

- Python 3.10+ (with special focus on free-threaded builds: 3.13t, 3.14t)
- [uv](https://github.com/astral-sh/uv) package manager (required)

## Setup Development Environment

```bash
# Clone the repo
git clone https://github.com/datologyai/zephon.git
cd zephon

# Install dependencies (dev + test groups)
make setup
```

## Running Tests

```bash
# Run unit tests
make test

# Run integration tests
make integration

# Pass extra args (e.g., verbose, specific file)
make test EXTRA_ARGS="-v tests/zephon/api/"
```

## Code Quality

```bash
# Check formatting, linting, and types (no changes)
make lint

# Auto-format and fix issues
make format
```

## Make Targets Summary

| Target | Description |
|--------|-------------|
| `make setup` | Install dev/test dependencies via uv |
| `make test` | Run unit tests |
| `make integration` | Run integration tests |
| `make lint` | Check formatting, linting, and types |
| `make format` | Auto-fix formatting and lint issues |

## Running Examples

```bash
uv run python examples/run_basic.py      # Basic in-memory pipeline
uv run python examples/run_jsonl.py      # JSONL file reading
uv run python examples/run_mixture.py    # Dataset mixture
```

## Project Structure

```
zephon/
├── api/          # User-facing Pipeline builder
├── core/         # Engine, graph, planner, constants
├── io/           # Datasets, storage backends, formats
├── ops/          # Operators (batch, tokenize, shuffle, etc.)
├── runners/      # Stage execution (inline, threads, process)
├── work/         # WorkSource and chunk management
└── observability/# Metrics and stats

tests/
├── integration/  # End-to-end tests (use make integration)
└── zephon/       # Unit tests mirroring source structure
```

## Key Reading

- [sample_lifecycle.md](sample_lifecycle.md) - Essential architecture documentation covering sample flow, determinism, checkpointing, and operator contracts
