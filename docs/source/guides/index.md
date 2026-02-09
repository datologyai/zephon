# Guides

This section contains detailed guides for using and developing with Zephon.

```{toctree}
:maxdepth: 2

dev_guide
prefetch_op
sample_lifecycle
```

## Available Guides

### [Developer Guide](dev_guide.md)

Getting started with Zephon development:
- Setting up your development environment
- Running tests
- Code quality tools
- Project structure

### [Prefetch Operator Guide](prefetch_op.md)

Optimizing data loading latency:
- How prefetching works
- Configuration options
- When to use prefetch
- Performance monitoring

### [Sample Lifecycle](sample_lifecycle.md)

Deep dive into Zephon's internals:
- Sample flow through the pipeline
- Determinism guarantees
- Checkpoint/restore behavior
- Operator contracts for fan-out, packing, and filtering
