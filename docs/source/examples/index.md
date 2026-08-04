# Examples

This section contains complete, runnable examples demonstrating Zephon's features.

## Basic Examples

### Your First Pipeline

The file-backed tokenization and batching example used by the Quick Start guide:

```{literalinclude} ../../../examples/your_first_pipeline.py
:language: python
:caption: examples/your_first_pipeline.py
```

### Basic Pipeline

A minimal example showing pipeline construction and iteration:

```{literalinclude} ../../../examples/run_basic.py
:language: python
:caption: examples/run_basic.py
```

### JSONL Reading

Reading data from JSONL files:

```{literalinclude} ../../../examples/run_jsonl.py
:language: python
:caption: examples/run_jsonl.py
```

### Dataset Mixture

Mixing multiple datasets with different weights:

```{literalinclude} ../../../examples/run_mixture.py
:language: python
:caption: examples/run_mixture.py
```

## Advanced Examples

### Prefetching

Using prefetch for remote data with reduced latency:

```{literalinclude} ../../../examples/run_with_prefetch.py
:language: python
:caption: examples/run_with_prefetch.py
```

## Running the Examples

All examples can be run directly with `uv`:

```bash
# Quick Start pipeline
uv run python examples/your_first_pipeline.py

# Basic pipeline
uv run python examples/run_basic.py

# JSONL reading
uv run python examples/run_jsonl.py

# Dataset mixture
uv run python examples/run_mixture.py

# Prefetching
uv run python examples/run_with_prefetch.py
```
