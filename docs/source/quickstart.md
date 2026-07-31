# Quick Start

This guide will help you get started with Zephon in a few minutes.

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install zephon
```

Or using pip:

```bash
pip install zephon
```

For cloud storage support:

```bash
uv pip install "zephon[cloud]"
# or: pip install "zephon[cloud]"
```

For specific data formats:

```bash
# Parquet files
pip install "zephon[parquet]"

# MosaicML Streaming/MDS format
pip install "zephon[streaming]"

# LitData format
pip install "zephon[litdata]"

# Vortex format (Python 3.11+ only)
pip install "zephon[vortex]"
```

## Basic Concepts

Zephon pipelines have three main components:

1. **Dataset**: Defines where your data lives and how to read it
2. **WorkSource**: Controls how samples are distributed across workers
3. **Pipeline**: Chains together operators to transform your data

## Your First Pipeline

Here's a minimal example using in-memory data:

```python
from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# 1. Create some sample data
shards = {
    0: InMemoryShard([
        {"text": "Hello world"},
        {"text": "Zephon is fast"},
        {"text": "Data loading made easy"},
    ])
}

# 2. Create a dataset from the shards
ds = Dataset.from_dict("demo", shards)

# 3. Create a work source that controls sample distribution
ws = StaticMixtureWorkSource(
    datasets=[ds],
    mixture=MixtureSpec({"demo": 1.0}),
    chunk_size=1,
    seed=42,
)

# 4. Build the pipeline
pipeline = (
    Pipeline(ws)
    .decode_text()  # Extract text from samples
    .tokenize(
        tokenizer_id="__fallback__",
        field="text",
        parallelism=1,
        preserve_upstream_payload=True,  # Keep "text" field for display
    )
    .batch(
        microbatch_size=2, drop_last=False
    )  # Group into batches; keep last partial batch
)

# 5. Iterate over batches
batch_count = 0
for batch_num, batch in enumerate(pipeline):
    training_batch = batch.to_training(dtype=None)  # Use Python lists (not numpy/torch) to avoid requiring torch
    texts = training_batch["texts"]
    batch_count += 1
    print(f"   Batch {batch_num}: {texts}")
```

You can run that first pipeline with the following commands:
```shell
cd zephon
uv run python examples/your_first_pipeline.py

...

============================================================
Your First Zephon Pipeline
============================================================

1. Creating sample data (3 records in a single shard)...
   Shard 0 contains 3 samples

2. Creating a Dataset from the shards...
   Dataset 'demo' has 3 total samples

3. Creating a StaticMixtureWorkSource...
   Work source configured with mixture={'demo': 1.0}, chunk_size=1, seed=42

4. Building the pipeline (decode_text -> tokenize -> batch with microbatch_size=2)...
   Pipeline ready. decode_text() extracts the 'text' field; tokenize() adds input_ids; batch() groups 2 samples.

5. Iterating over batches:
------------------------------------------------------------
   Batch 0: ['Hello world', 'Zephon is fast']
   Batch 1: ['Data loading made easy']
------------------------------------------------------------

Done! You processed 3 samples in 2 batch(es).
```

## Reading from Files

For real workloads, you'll typically read from files. Here's an example with JSONL:

```python
from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# Point to your JSONL directory (auto-detects format)
ds = Dataset.from_path(
    name="my_data",
    path="data/train/",  # Directory containing *.jsonl files
)

ws = StaticMixtureWorkSource(
    datasets=[ds],
    mixture=MixtureSpec({"my_data": 1.0}),
    chunk_size=64,
    seed=42,
)

pipeline = (
    Pipeline(ws)
    .decode_text()
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=8)
)

for batch in pipeline:
    # Use batch for training
    pass
```

## Adding Tokenization

Zephon includes built-in tokenization support:

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
