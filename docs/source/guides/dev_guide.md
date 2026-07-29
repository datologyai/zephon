# Zephon Developer Guide

## What is Zephon?

Zephon is a high-performance, modular, multimodal-first data loader for PyTorch and ML applications. It provides:

- Fluent builder-pattern API for composing data pipelines
- Deterministic checkpoint/restart for fault-tolerant training
- Cloud storage support (S3, GCS, local filesystem)
- Elastic training (change rank count while preserving order)
- Multiple data formats (JSONL, LitData, MosaicML Streaming/MDS, Vortex)

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

# Install pre-commit hooks (runs ruff format + ruff check before each commit)
pre-commit install
```

## Running Tests

```bash
# Run unit tests
make test

# Run integration tests
make integration

# Pass extra args (e.g., verbose, specific file)
make test EXTRA_ARGS="-v tests/zephon/test_pipeline.py"
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

## Optional Format Dependencies

Some data formats require additional dependencies. Install them using the optional dependency groups defined in `pyproject.toml`:

```bash
# For Vortex format support (Python 3.11+ only)
uv pip install -e ".[vortex]"

# For MosaicML Streaming/MDS format support
uv pip install -e ".[streaming]"

# For LitData format support
uv pip install -e ".[litdata]"

# For Parquet format support
uv pip install -e ".[parquet]"

# For cloud storage (S3 and GCS)
uv pip install -e ".[cloud]"

# Combine multiple extras
uv pip install -e ".[vortex,streaming,cloud]"
```

## Publishing Releases

Zephon is published to the private DatologyAI package index on AWS CodeArtifact
(domain `datologyai`, repository `main`, `us-east-1`). The project uses
`setuptools-scm` for version management, which derives versions from git tags.

### Publishing an Alpha/Internal Release

1. **Create a version tag** following [PEP 440](https://peps.python.org/pep-0440/):
   ```bash
   git tag v0.1.0a1  # alpha release
   git tag v0.1.0    # stable release
   ```

2. **Mint a CodeArtifact token** (needs AWS credentials with publish access; SSO
   works):
   ```bash
   export UV_PUBLISH_USERNAME=aws
   export UV_PUBLISH_PASSWORD=$(aws codeartifact get-authorization-token \
     --domain datologyai --domain-owner 764487710063 --region us-east-1 \
     --query authorizationToken --output text)
   ```

3. **Build and publish** to CodeArtifact (`--index datologyai` uses the
   `publish-url` from `pyproject.toml`):
   ```bash
   uv build
   uv publish --index datologyai dist/*
   ```

4. **Push the tag**:
   ```bash
   git push origin v0.1.0a1
   ```

### Installing Internal Releases

Reads from CodeArtifact use HTTP token auth. Mint a token (needs AWS credentials;
SSO works) and expose it to `uv` via the named-index env vars, then install from
the `datologyai` index:

```bash
export UV_INDEX_DATOLOGYAI_USERNAME=aws
export UV_INDEX_DATOLOGYAI_PASSWORD=$(aws codeartifact get-authorization-token \
  --domain datologyai --domain-owner 764487710063 --region us-east-1 \
  --query authorizationToken --output text)
INDEX=datologyai=https://datologyai-764487710063.d.codeartifact.us-east-1.amazonaws.com/pypi/main/simple/

# Install a specific version
uv pip install zephon==0.0.1a1 --index "$INDEX"

# Install latest (including pre-releases)
uv pip install zephon --pre --index "$INDEX"
```

### Version Scheme

- `X.Y.ZaN` - Alpha releases (e.g., `0.1.0a1`)
- `X.Y.ZbN` - Beta releases (e.g., `0.1.0b1`)
- `X.Y.ZrcN` - Release candidates (e.g., `0.1.0rc1`)
- `X.Y.Z` - Stable releases (e.g., `0.1.0`)

Note: If you build with uncommitted changes, setuptools-scm will append a `.devN+gHASH` suffix to indicate a dirty build.

## Key Reading

- [sample_lifecycle.md](sample_lifecycle.md) - Essential architecture documentation covering sample flow, determinism, checkpointing, and operator contracts
