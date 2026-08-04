# Quick Start

This guide will help you get started with Zephon in a few minutes.

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install zephon

# Hugging Face tokenizer support used by the example below
uv pip install transformers
```

For cloud storage support:

```bash
uv pip install "zephon[cloud]"
```

For specific data formats:

```bash
# Parquet files
uv pip install "zephon[parquet]"

# MosaicML Streaming/MDS format
uv pip install "zephon[streaming]"

# LitData format
uv pip install "zephon[litdata]"

# Vortex format (Python 3.11+ only)
uv pip install "zephon[vortex]"
```

To run the checked-in examples or contribute to Zephon, install from a source
checkout instead:

```bash
git clone https://github.com/datologyai/zephon.git
cd zephon
uv sync --group dev --group test
```

## Basic Concepts

Zephon pipelines have three main components:

1. **Dataset**: Defines where your data lives and how to read it
2. **WorkSource**: Declares the curriculum: datasets, proportions, and ordering
3. **Pipeline**: Chains together operators to transform your data

## Your First Pipeline

Here's a representative text-training pipeline. Replace the path with a
directory containing your JSONL shards:

```python
from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# 1. Discover the dataset's shards
dataset = Dataset.from_path("train", "data/train/")

# 2. Declare the sample mixture and ordering
ws = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"train": 1.0}),
    seed=42,
)

# 3. Decode, tokenize, and batch for training
pipeline = (
    Pipeline(ws)
    .decode_text()
    .tokenize(
        tokenizer_id="gpt2",
        field="text",
        padding=True,
        parallelism=1,
        preserve_upstream_payload=True,  # Keep "text" field for display
    )
    .batch(microbatch_size=2, drop_last=False)
)

for batch in pipeline:
    training_batch = batch.to_training()
    print(training_batch["texts"])
    train_step(training_batch)
```

From a source checkout, run the tested version of this example with:

```shell
uv run python examples/your_first_pipeline.py
```

## Adjusting Tokenization

Tokenization can be tuned per workload:

```python
pipeline = (
    Pipeline(ws)
    .decode_text()
    .tokenize(
        tokenizer_id="gpt2",  # Any HuggingFace tokenizer
        field="text",
        max_length=512,
        parallelism=4,        # Parallel tokenization workers
    )
    .batch(microbatch_size=8)
)
```

## Pipeline Options

Configure pipeline behavior with `.options()`:

```python
pipeline = pipeline.options(
    default_stage_prefetch=2,  # Prefetch between stages
    prefetch_batches=3,        # Prefetch final batches
)
```

## Checkpointing

Zephon supports deterministic checkpointing for fault-tolerant training:

```python
# Save checkpoint
state = pipeline.checkpoint()

# Later, restore from checkpoint
pipeline.restore(state)

# Continue iterating - samples resume exactly where you left off
for batch in pipeline:
    ...
```

## Next Steps

- Check out the [Understanding Zephon](understanding/worksources.md) section to learn more about Zephon's concepts and what's happening behind the scenes
- See the [API Reference](api/zephon_pipeline) for detailed documentation
- Check out [Examples](examples/index) for more complete examples
