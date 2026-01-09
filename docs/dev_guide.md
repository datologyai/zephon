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

## Publishing Releases

Zephon is published to our internal pyx.dev registry. The project uses `setuptools-scm` for version management, which derives versions from git tags.

### Publishing an Alpha/Internal Release

1. **Create a version tag** following [PEP 440](https://peps.python.org/pep-0440/):
   ```bash
   git tag v0.1.0a1  # alpha release
   git tag v0.1.0    # stable release
   ```

2. **Build the package** (ensure no uncommitted changes for a clean version):
   ```bash
   uv build
   ```

3. **Publish to the internal registry**:
   ```bash
   # Dry run first to validate
   uv publish dist/* --publish-url https://api.pyx.dev/v1/upload/datologyai/main --dry-run

   # Publish for real
   uv publish dist/* --publish-url https://api.pyx.dev/v1/upload/datologyai/main
   ```

4. **Push the tag**:
   ```bash
   git push origin v0.1.0a1
   ```

### Installing Internal Releases

```bash
# Install a specific version
uv pip install zephon==0.0.1a1 --index https://api.pyx.dev/simple/datologyai/main

# Install latest (including pre-releases)
uv pip install zephon --pre --index https://api.pyx.dev/simple/datologyai/main
```

### Version Scheme

- `X.Y.ZaN` - Alpha releases (e.g., `0.1.0a1`)
- `X.Y.ZbN` - Beta releases (e.g., `0.1.0b1`)
- `X.Y.ZrcN` - Release candidates (e.g., `0.1.0rc1`)
- `X.Y.Z` - Stable releases (e.g., `0.1.0`)

Note: If you build with uncommitted changes, setuptools-scm will append a `.devN+gHASH` suffix to indicate a dirty build.

## Key Reading

- [sample_lifecycle.md](sample_lifecycle.md) - Essential architecture documentation covering sample flow, determinism, checkpointing, and operator contracts
