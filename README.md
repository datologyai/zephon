<h2><p align="center">Zephon</p></h2>
<p align="center"><em>An Elastically Deterministic, High-Performance Data Loader for Stateful Foundation Model Pipelines</em></p>

<p align="center">
  <a href="https://github.com/datologyai/zephon/actions/workflows/pytest.yaml"><img src="https://github.com/datologyai/zephon/actions/workflows/pytest.yaml/badge.svg" alt="Unit Tests" /></a>
  <a href="https://github.com/datologyai/zephon/actions/workflows/pytest-integration.yaml"><img src="https://github.com/datologyai/zephon/actions/workflows/pytest-integration.yaml/badge.svg" alt="Integration Tests" /></a>
  <a href="https://github.com/datologyai/zephon/actions/workflows/linting.yaml"><img src="https://github.com/datologyai/zephon/actions/workflows/linting.yaml/badge.svg" alt="Linting" /></a>
  <a href="https://datologyai.github.io/zephon/"><img src="https://img.shields.io/badge/docs-github%20pages-blue.svg" alt="Docs" /></a>
  <a href="https://github.com/datologyai/zephon/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License" /></a>
</p>

---

Zephon is a high-performance data loading library that separates **what** you train on from **how** data is loaded and transformed. It supports online processing (tokenization, sequence packing, filtering, dynamic mixtures) during training rather than requiring expensive offline preprocessing, while guaranteeing deterministic, reproducible output regardless of parallelism or hardware topology.

### ✨ Key Features

- 🔁 **Elastic determinism** -- checkpoint on 8 GPUs, resume on 4 (or 32). The global sample order is preserved across topology changes. Execution parameters (parallelism, runner type, micro-batch size) are pure performance knobs that never change data order.
- ⚡ **Online processing** -- tokenize, pack sequences, filter, and mix datasets on the fly. No separate preprocessing step needed.
- 📂 **Format-agnostic** -- automatic format detection for JSONL, Parquet, MosaicML Streaming/MDS, LitData, and Vortex.
- ☁️ **Cloud-native** -- stream from S3, GCS, or Azure with optional prefetching that warms a local cache ahead of consumption. Local and HPC distributed filesystems work out of the box as well.
- 🔗 **Composable pipeline API** -- fluent builder with built-in operators for tokenization, batching, shuffling, sequence packing, and mixture enforcement.
- 💾 **Deterministic checkpointing** -- save and restore full pipeline state for fault-tolerant training with chunk-level granularity.
- 🧩 **Pluggable runners** -- run stages inline, in threads, in processes, or on Ray (experimental).
- 🔥 **Framework-agnostic** -- your data pipeline shouldn't be coupled to your training framework. Zephon pipelines are plain Python iterables that work with PyTorch, JAX, TensorFlow, or anything else, with first-class integrations for `torch.DataLoader` and `torchdata.StatefulDataLoader`.

---

## 🚀 Quick Start

### Installation

```bash
uv pip install zephon

# Hugging Face tokenizer support used by the example below
uv pip install transformers

# With cloud storage support (S3, GCS, Azure)
uv pip install "zephon[cloud]"

# Format-specific extras
uv pip install "zephon[parquet]"
uv pip install "zephon[streaming]"
uv pip install "zephon[litdata]"
```

### Your First Pipeline

```python
from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# 1. Point Zephon at your training data
dataset = Dataset.from_path("train", "/data/train/")

# 2. Declare what to train on
ws = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"train": 1.0}),
    seed=42,
)

# 3. Build the processing pipeline
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

# 4. Iterate
for batch in pipeline:
    training_batch = batch.to_training()
    print(training_batch["texts"])
    train_step(training_batch)
```

### Mixing Datasets

```python
fineweb = Dataset.from_path("fineweb", "/data/fineweb")
dclm = Dataset.from_path("dclm", "/data/dclm")

ws = StaticMixtureWorkSource(
    datasets=[fineweb, dclm],
    mixture=MixtureSpec({"fineweb": 0.7, "dclm": 0.3}),
    chunk_size=16384,
    seed=42,
)
```

---

## 🏗️ How It Works

Zephon pipelines follow a three-level architecture:

```
WorkSource          what to train on (datasets, mixtures, shuffling)
    │               produces lightweight pointers: (dataset, shard, sample)
    ▼
Pipeline            how to process (tokenize, pack, batch, ...)
    │               composable chain of operators
    ▼
Engine              where and when to execute (threads, processes, queues)
                    deterministic scheduling with backpressure
```

The **WorkSource** generates a deterministic sequence of sample pointers without doing any I/O. The **Pipeline** chains operators that transform raw data into training batches. The **Engine** compiles the pipeline into concurrent stages connected by bounded queues, overlapping data fetching, processing, and GPU training.

Use `pipeline.explain()` to inspect the compiled execution plan:

```
Stage[0] runner=threads cap=8   ops=['fetch@p4', 'tokenize@p4']
Stage[1] runner=inline  cap=1   ops=['ensure_mixture@p1', 'batch@p1']
  ==[final_prefetch=3]==> pipeline_end
```

### Checkpointing

```python
# Save
state = pipeline.checkpoint()

# Restore (works across different GPU counts)
pipeline.restore(state)
for batch in pipeline:
    ...
```

---

## 📖 Documentation

Full documentation is available at [datologyai.github.io/zephon](https://datologyai.github.io/zephon), including:

- [Quick Start](https://datologyai.github.io/zephon/quickstart.html) -- get running in 5 minutes
- [Basic Concepts](https://datologyai.github.io/zephon/basic_concepts.html) -- WorkSources, Pipelines, Operators, Runners, and the Engine
- [Elastic Determinism](https://datologyai.github.io/zephon/understanding/determinism.html) -- reproducibility guarantees across GPU counts
- [Checkpointing](https://datologyai.github.io/zephon/understanding/checkpointing.html) -- fault-tolerant training with mid-epoch resumption
- [API Reference](https://datologyai.github.io/zephon/api/zephon_pipeline.html) -- full Pipeline and operator reference

## 📄 License

Zephon is released under the [Apache 2.0 License](LICENSE).
